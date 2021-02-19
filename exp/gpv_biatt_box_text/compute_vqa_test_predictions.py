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
from .models.gpv import GPV
from .models.losses import GPVCriterion
from datasets.coco_datasets import CocoVqaTestOriginalSplitDataset
from utils.bbox_utils import vis_bbox
from utils.detr_misc import collate_fn
import utils.io as io
from utils.html_writer import HtmlWriter


def make_predictions(model,dataloader,samples,cfg):
    eval_dir = os.path.join(cfg.exp_dir,'eval')
    results = []
    cnt = 0
    detokenizer = TreebankWordDetokenizer()
    model.eval()
    for i,data in enumerate(tqdm(dataloader)):
        if (cfg.eval.num_eval_batches is not None) \
            and (i > cfg.eval.num_eval_batches):
            break

        imgs, queries = data
        imgs = imgs.to(torch.device(cfg.gpu))

        outputs = model(imgs,queries,None)
        topk_answers = torch.topk(outputs['answer_logits'][-1],k=1,dim=-1)
        topk_answer_ids = topk_answers.indices.detach().cpu().numpy()
        pred_answers = model.token_ids_to_words(topk_answer_ids[:,:,0])
        B = imgs.size(0)
        for b in range(B):
            answer = []
            for token in pred_answers[b]:
                if token in ['__stop__','__pad__']:
                    break
                answer.append(token)
            answer = detokenizer.detokenize(answer)

            result = {
                'question_id': samples[cnt]['question_id'],
                'answer': answer
            }
            results.append(result)
            cnt += 1

    io.dump_json_object(
        predictions,
        os.path.join(
            eval_dir,
            f'{cfg.eval.task}_{cfg.eval.subset}_predictions.json'))


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


@hydra.main(config_path=f'../../configs',config_name=f"exp/gpv_biatt_box_text_coco")
def main(cfg):
    eval_dir = os.path.join(cfg.exp_dir,'eval')
    io.mkdir_if_not_exists(eval_dir,recursive=True)
    print(cfg.pretty())
    print(cfg.exp_dir)
    dataset = CocoVqaTestOriginalSplitDataset(
        cfg.task_configs,cfg.eval.subset)
    dataloader = dataset.get_dataloader(
        batch_size=cfg.eval.batch_size,
        num_workers=cfg.eval.num_workers,
        pin_memory=True,
        shuffle=False)
    samples = dataset.samples

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


if __name__=='__main__':
    main()

