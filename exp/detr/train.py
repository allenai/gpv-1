import os
import hydra
import torch
import itertools
import numpy as np
import skimage.io as skio

from .models.detr import create_detr
from datasets.clevr_detection_train_task import ClevrDetectionTrainTask
from utils.set_criterion import SetCriterion
from utils.matcher import HungarianMatcher
from utils.bbox_utils import vis_bbox
import utils.io as io


def train_model(model,dataloader,cfg):
    backbone_params = [p for n, p in model.named_parameters() \
            if 'backbone' in n and p.requires_grad]
    other_params = [p for n, p in model.named_parameters() \
            if 'backbone' not in n and p.requires_grad]
    param_dicts = [
        {'params': other_params},
        {'params': backbone_params, 'lr': cfg.model.detr.lr_backbone}]
    optimizer = torch.optim.AdamW(
        param_dicts, 
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        cfg.training.lr_drop)

    wts = cfg.training.loss_wts
    wts_dict = {
        'loss_ce': wts.ce,
        'loss_bbox': wts.bbox,
        'loss_giou': wts.giou,
    }
    aux_wts_dict = {}
    for i in range(cfg.model.detr.num_decoder_layers - 1):
        aux_wts_dict.update({f'{k}_{i}': v for k, v in wts_dict.items()})
    
    wts_dict.update(aux_wts_dict)

    matcher = HungarianMatcher(
        cost_class=cfg.training.cost_wts.ce,
        cost_bbox=cfg.training.cost_wts.bbox,
        cost_giou=cfg.training.cost_wts.giou).cuda()

    set_criterion = SetCriterion(
        num_classes=cfg.model.detr.num_classes,
        matcher=matcher,
        weight_dict=wts_dict,
        eos_coef=cfg.training.eos_coef,
        losses=['labels','boxes','cardinality']).cuda()

    step = 0
    for epoch in range(cfg.training.num_epochs):
        for it,data in enumerate(dataloader):
            imgs, targets = data
            imgs = imgs.to(torch.device(0))
            targets = [{k: v.cuda() for k, v in t.items()} for t in targets]
            
            model.train()
            set_criterion.train()
            
            outputs = model(imgs)
            losses = set_criterion(outputs,targets)
            

            loss_to_optim = sum(
                losses[k] * wts_dict[k] for k in losses.keys() if k in wts_dict)
            
            if step%10==0:
                loss_str = f'Epoch: {epoch} | Iter: {it} | Step: {step} | '
                for k in itertools.chain(
                        wts_dict.keys(),['class_error','cardinality_error']):
                    loss_value = round(losses[k].item(),4)
                    loss_str += f'{k}: {loss_value} | '
                print(loss_str)


            if step%100==0:
                with torch.no_grad():
                    model.eval()
                    visualize(model,dataloader,cfg,step)
                    model.train()
                    
            optimizer.zero_grad()
            loss_to_optim.backward()
            if cfg.training.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    cfg.training.clip_max_norm)
            optimizer.step()

            step += 1

        lr_scheduler.step()


def visualize(model,dataloader,cfg,step):
    vis_dir = os.path.join(cfg.exp_dir,'train_vis')
    io.mkdir_if_not_exists(vis_dir,recursive=True)

    count = 0
    for data in dataloader:
        imgs, targets = data
        imgs = imgs.to(torch.device(0))

        outputs = model(imgs)
        imgs = 255*(0.5+0.25*imgs.tensors.permute(0,2,3,1).detach().cpu().numpy())
        imgs = imgs.astype(np.uint8)

        # visualize predictions
        pred_prob = outputs['pred_logits'].softmax(-1)
        topk = torch.topk(pred_prob[:,:,0],k=5,dim=1)
        topk_ids = topk.indices.detach().cpu().numpy()
        pred_boxes = outputs['pred_boxes'].detach().cpu().numpy()
        gt_boxes = [t['boxes'].detach().numpy() for t in targets]
        B = pred_boxes.shape[0]
        for b in range(B):
            if count+b >= cfg.training.num_vis_samples:
                break

            # visualize prediction
            boxes = pred_boxes[b,topk_ids[b]]
            print(boxes)
            vis_img = imgs[b]
            for k in range(boxes.shape[0]):
                vis_bbox(boxes[k],vis_img,modify=True)

            # visualize gt
            boxes = gt_boxes[b]
            #vis_img = img[b]
            for k in range(boxes.shape[0]):
                vis_bbox(boxes[k],vis_img,color=(0,255,0),modify=True)

            fname = os.path.join(
                vis_dir,
                str(step).zfill(6) + '_' + str(count+b).zfill(4) + '.png')
            skio.imsave(fname,vis_img)

        count += B


@hydra.main(config_path=f'../../configs',config_name=f"exp/detr")
def main(cfg):
    print(cfg.pretty())

    model = create_detr(cfg.model.detr).cuda()
    
    dataset = ClevrDetectionTrainTask(cfg.task.clevr_detection_train,'train')
    dataloader = dataset.get_dataloader(batch_size=cfg.training.batch_size,num_workers=8)

    train_model(model,dataloader,cfg)


if __name__=='__main__':
    main()

