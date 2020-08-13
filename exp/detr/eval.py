import os
import hydra
import torch
import itertools
import numpy as np
import skimage.io as skio
from tqdm import tqdm

from .models.detr import create_detr
from datasets.clevr_detection_train_task import ClevrDetectionTrainTask
from utils.bbox_utils import vis_bbox
import utils.io as io

import third_party.detection_metrics.lib.Evaluator as evaluator

def eval_model(model,dataloader,cfg):
    model.eval()
    cnt = 0
    all_boxes = evaluator.BoundingBoxes()
    for it,data in enumerate(tqdm(dataloader)):
        imgs, targets = data
        imgs = imgs.to(torch.device(0))

        outputs = model(imgs)
        pred_prob = outputs['pred_logits'].softmax(-1).detach().cpu().numpy()
        pred_boxes = outputs['pred_boxes'].detach().cpu().numpy()
        B = pred_boxes.shape[0]
        H = imgs.tensors.size(2)
        W = imgs.tensors.size(3)
        img_size = (W,H)
    
        for b in range(B):
            img_name = str(cnt).zfill(6)

            gt_boxes = targets[b]['boxes'].numpy()
            for k in range(gt_boxes.shape[0]):
                x,y,w,h = gt_boxes[k]
                x = x - 0.5*w
                y = y - 0.5*h
                all_boxes.addBoundingBox(evaluator.BoundingBox(
                    imageName=img_name,
                    classId='relevant',
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    typeCoordinates=evaluator.CoordinatesType.Relative,
                    imgSize=img_size,
                    bbType=evaluator.BBType.GroundTruth,
                    format=evaluator.BBFormat.XYWH))
            
            boxes = pred_boxes[b]
            probs = pred_prob[b]
            for k in range(boxes.shape[0]):
                x,y,w,h = boxes[k]
                x = x - 0.5*w
                y = y - 0.5*h
                all_boxes.addBoundingBox(evaluator.BoundingBox(
                    imageName=img_name,
                    classId='relevant',
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    typeCoordinates=evaluator.CoordinatesType.Relative,
                    imgSize=img_size,
                    bbType=evaluator.BBType.Detected,
                    classConfidence=probs[k][0],
                    format=evaluator.BBFormat.XYWH))

            cnt+=1

    eval_engine = evaluator.Evaluator()
    metrics = eval_engine.GetPascalVOCMetrics(all_boxes,cfg.eval.iou_thresh)
    return metrics


@hydra.main(config_path=f'../../configs',config_name=f"exp/detr")
def main(cfg):
    print(cfg.pretty())

    model = create_detr(cfg.model.detr).cuda()

    ckpt_pth = os.path.join(cfg.ckpt_dir,str(cfg.eval.step).zfill(6)+'.pth')
    model.load_state_dict(torch.load(ckpt_pth)['model'])
    
    dataset = ClevrDetectionTrainTask(cfg.task.clevr_detection_train,'train')
    
    dataloader = dataset.get_dataloader(
        batch_size=cfg.eval.batch_size,
        num_workers=8)

    metrics = eval_model(model,dataloader,cfg)


if __name__=='__main__':
    main()

