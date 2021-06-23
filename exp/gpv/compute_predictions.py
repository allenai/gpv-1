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
from data.web.synonyms import WEB_SYNONYMS
from . import evaluators
from .models.gpv import GPV
from .models.losses import GPVCriterion
from datasets.coco_multitask_dataset import CocoMultitaskDataset
from utils.bbox_utils import vis_bbox
from utils.detr_misc import collate_fn
import utils.io as io
from utils.html_writer import HtmlWriter

# Hacky, please overlook for now
import json
web_vocab = json.load(open('/home/amitak/gpv-1/data/learning_phase_data/vocab/all_queries_verified.json'))
WEB_SYNONYMS.update({k: [k] for k in web_vocab if k not in WEB_SYNONYMS})
WEB_SYNONYMS.update({w: [w] for k in web_vocab for w in k.split() if w not in WEB_SYNONYMS})

def make_predictions(model,dataloader,samples,cfg):
    vocab_mask=None
    if cfg.eval.task=='CocoClassification':
        tokens,vocab_mask = create_coco_vocab_mask(model)
        vocab_mask = torch.FloatTensor(vocab_mask).cuda(cfg.gpu)
    elif cfg.eval.task=='WebQa':
        tokens,vocab_mask = create_web_vocab_mask(model)
        vocab_mask = torch.FloatTensor(vocab_mask).cuda(cfg.gpu)
    
    eval_dir = os.path.join(cfg.exp_dir,'eval')
    boxes_h5py = h5py.File(os.path.join(
        eval_dir,f'{cfg.eval.task}_{cfg.eval.subset}_boxes.h5py'),'w')
    task_id_name = evaluators.task_to_id[cfg.eval.task]
    predictions = {}
    cnt = 0
    detokenizer = TreebankWordDetokenizer()
    model.eval()
    for i,data in enumerate(tqdm(dataloader)):
        if (cfg.eval.num_eval_batches is not None) \
            and (i > cfg.eval.num_eval_batches):
            break

        imgs, queries, targets = data
        imgs = imgs.to(torch.device(cfg.gpu))

        outputs = model(imgs,queries,None,vocab_mask=vocab_mask)
        relevance = outputs['pred_relevance_logits'].softmax(-1).detach().cpu().numpy()
        pred_boxes = outputs['pred_boxes'].detach().cpu().numpy()
        topk_answers = torch.topk(outputs['answer_logits'][-1],k=1,dim=-1)
        topk_answer_ids = topk_answers.indices.detach().cpu().numpy()
        pred_answers = model.token_ids_to_words(topk_answer_ids[:,:,0])
        for b in range(len(targets)):
            scores, boxes = zip(*sorted(zip(
                relevance[b,:,0].tolist(),pred_boxes[b].tolist()),
                key=lambda x: x[0],reverse=True))
            scores = np.array(scores,dtype=np.float32)
            boxes = np.array(boxes,dtype=np.float32)
            sample_id = samples[cnt][task_id_name]
            answer = []
            for token in pred_answers[b]:
                if token in ['__stop__','__pad__']:
                    break
                answer.append(token)
            answer = detokenizer.detokenize(answer)

            predictions[sample_id] = {
                'answer': answer
            }
            grp = boxes_h5py.create_group(str(sample_id))
            grp.create_dataset('boxes',data=boxes)
            grp.create_dataset('relevance',data=scores)
            cnt += 1

    boxes_h5py.close()
    io.dump_json_object(
        predictions,
        os.path.join(
            eval_dir,
            f'{cfg.eval.task}_{cfg.task_configs.data_split}_{cfg.eval.subset}_predictions.json'))


def create_coco_vocab_mask(model,use_syns=False):
    L = len(model.vocab)
    mask = -10000*np.ones([L],dtype=np.float32)
    tokens = []
    for coco_cls in SYNONYMS:
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


def create_web_vocab_mask(model,use_syns=False):
    L = len(model.vocab)
    mask = -10000*np.ones([L],dtype=np.float32)
    tokens = []
    for web_ans in WEB_SYNONYMS:
        syns = [web_ans]
        if use_syns is True:
            syns = WEB_SYNONYMS[web_ans]
        
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


@hydra.main(config_path=f'../../configs',config_name=f"exp/gpv")
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
        model = GPV(cfg.model)
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
        eval_dir,f'{cfg.eval.task}_{cfg.task_configs.data_split}_{cfg.eval.subset}_predictions.json'))

    boxes_h5py = h5py.File(os.path.join(
        eval_dir,f'{cfg.eval.task}_{cfg.task_configs.data_split}_{cfg.eval.subset}_boxes.h5py'),'r')

    if cfg.eval.task in ['CocoDetection','RefCocop']:
        samples = update_samples_with_image_size(
            cfg.task_configs.image_dir,
            samples)
    
    Evaluator = getattr(evaluators,cfg.eval.task)
    evaluator = Evaluator(samples,predictions,boxes_h5py)
    metrics = {}
    if cfg.eval.task=='RefCocop':
        metrics['everything'] = evaluator.evaluate()
    else:
        novelty_types = ['everything','seen_concepts','held_out_concepts']
        if cfg.task_configs.data_split=='original_split':
            novelty_types = ['everything']
        
        for novelty in novelty_types:
            metrics[novelty] = evaluator.evaluate(novelty)

    io.dump_json_object(
        metrics,
        os.path.join(
            eval_dir,
            f'{cfg.eval.task}_{cfg.task_configs.data_split}_{cfg.eval.subset}_metrics.json'))

    boxes_h5py.close()

if __name__=='__main__':
    main()

