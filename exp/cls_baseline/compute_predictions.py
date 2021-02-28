import os
import h5py
import hydra
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
import itertools
import numpy as np
import skimage.io as skio
import imagesize
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from utils.detr_misc import collate_fn as detr_collate_fn
from torch.utils.data.dataloader import default_collate

from data.coco.synonyms import SYNONYMS
import exp.gpv_box_text.evaluators as evaluators
from .train import Classifier
from datasets.coco_multitask_dataset import CocoMultitaskDataset
from utils.bbox_utils import vis_bbox
from utils.detr_misc import collate_fn
import utils.io as io
from utils.html_writer import HtmlWriter


def make_predictions(model,dataloader,samples,cfg):
    labels = list(SYNONYMS.keys())
    eval_dir = os.path.join(cfg.exp_dir,'eval')
    task_id_name = evaluators.task_to_id[cfg.eval.task]
    predictions = {}
    cnt = 0
    model.eval()
    for i,data in enumerate(tqdm(dataloader)):
        if (cfg.eval.num_eval_batches is not None) \
            and (i > cfg.eval.num_eval_batches):
            break

        imgs, queries, targets = data
        imgs = imgs.to(torch.device(cfg.gpu))

        logits = model(imgs.tensors)
        preds = torch.argmax(logits,1).cpu().numpy()
        for b in range(preds.shape[0]):
            answer = labels[preds[b]]
            sample_id = samples[cnt][task_id_name]
            predictions[sample_id] = {
                'answer': answer
            }
            cnt += 1

    io.dump_json_object(
        predictions,
        os.path.join(
            eval_dir,
            f'{cfg.eval.task}_{cfg.eval.subset}_predictions.json'))


def create_coco_vocab_mask(model,use_syns=False):
    L = len(model.vocab)
    mask = -10000*np.ones([L],dtype=np.float32)
    tokens = []
    # categories = {
    #     "banana": 728,
    #     "baseball bat": 799,
    #     "bottle": 2912,
    #     "broccoli": 670,
    #     "donut": 523,
    #     "hot dog": 452,
    #     "keyboard": 750,
    #     "laptop": 1232,
    #     "train": 1281,
    #     "tv": 1231,
    #     "__cls__": 0,
    #     "__stop__": 0,
    #     "__pad__": 0}
    for coco_cls in SYNONYMS:
    #for coco_cls in categories:
        syns = [coco_cls]
        if use_syns is True:
            syns = SYNONYMS[coco_cls]
        
        for syn in syns:
            for token in word_tokenize(syn):
                if token in model.word_to_idx:
                    idx = model.word_to_idx[token]
                    mask[idx] = 0
                    tokens.append(token)
    
    for token in ['__stop__','__pad__']:
        idx = model.word_to_idx[token]
        mask[idx] = 0
        tokens.append(token)

    return tokens, mask
    

def update_samples_with_image_size(image_dir,samples):
    for sample in tqdm(samples):
        image_id = sample['image']['image_id']
        image_subset = sample['image']['subset']
        image_filename = os.path.join(
            os.path.join(image_dir,image_subset),
            'COCO_'+image_subset+'_'+str(image_id).zfill(12)+'.jpg')

        W,H = imagesize.get(image_filename)
        sample['image']['W'] = W
        sample['image']['H'] = H
    
    return samples


@hydra.main(config_path=f'../../configs',config_name=f"exp/cls_baseline")
def main(cfg):
    eval_dir = os.path.join(cfg.exp_dir,'eval')
    io.mkdir_if_not_exists(eval_dir,recursive=True)
    print(cfg.pretty())
    print(cfg.exp_dir)
    eval_task = cfg.eval.task
    learning_datasets = {eval_task:cfg.learning_datasets[eval_task]}
    dataset = CocoMultitaskDataset(
        learning_datasets,cfg.task_configs,cfg.eval.subset)
    dataloader = dataset.get_dataloader(
        batch_size=cfg.eval.batch_size,
        num_workers=cfg.eval.num_workers,
        pin_memory=True,
        shuffle=False)
    task_name = cfg.learning_datasets[eval_task].name
    samples = dataloader.dataset.datasets[task_name].samples

    if cfg.eval.predict is True:
        model = Classifier(cfg.model)
        model.cuda(cfg.gpu)
        loc = 'cuda:{}'.format(cfg.gpu)
        loaded_dict = torch.load(cfg.eval.ckpt, map_location=loc)['model']
        state_dict = model.state_dict()
        for k,v in state_dict.items():
            state_dict[k] = loaded_dict[f'module.{k}']
            state_dict[k].requires_grad = False
            
        model.load_state_dict(state_dict)

        with torch.no_grad():
            make_predictions(model,dataloader,samples,cfg)

    predictions = io.load_json_object(os.path.join(
        eval_dir,f'{cfg.eval.task}_{cfg.eval.subset}_predictions.json'))

    boxes_h5py = h5py.File(os.path.join(
        eval_dir,f'{cfg.eval.task}_{cfg.eval.subset}_boxes.h5py'),'r')

    if cfg.eval.task=='CocoDetection':
        samples = update_samples_with_image_size(
            cfg.task_configs.image_dir,
            samples)
    
    Evaluator = getattr(evaluators,cfg.eval.task)
    evaluator = Evaluator(samples,predictions,boxes_h5py)
    metrics = {}
    for novelty in ['everything','seen_concepts','held_out_concepts']:
        metrics[novelty] = evaluator.evaluate(novelty)

    io.dump_json_object(
        metrics,
        os.path.join(eval_dir,f'{cfg.eval.task}_{cfg.eval.subset}_metrics.json'))

    boxes_h5py.close()

if __name__=='__main__':
    main()

