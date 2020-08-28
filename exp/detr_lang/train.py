import os
import hydra
import torch
from torch.utils.tensorboard import SummaryWriter
import itertools
import numpy as np
import skimage.io as skio

from .models.gpv import GPV
from .eval import eval_model
from datasets.clevr_query_detection_train_task import ClevrQueryDetectionTrainTask
from utils.set_criterion import SetCriterion
from utils.matcher import HungarianMatcher
from utils.bbox_utils import vis_bbox
import utils.io as io
from utils.html_writer import HtmlWriter


def grad_norm(params):
    total_norm = 0
    for p in params:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    
    return total_norm ** (1. / 2)


def train_model(model,dataloaders,cfg):
    writer = SummaryWriter(log_dir=cfg.tb_dir)

    backbone_params = [p for n, p in model.named_parameters() \
            if 'backbone' in n and p.requires_grad]
    other_params = [p for n, p in model.named_parameters() \
            if 'backbone' not in n and 'bert.' not in n and p.requires_grad]
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

    if cfg.model.detr.aux_loss:
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
        for it,data in enumerate(dataloaders['train']):
            imgs, queries, targets = data
            imgs = imgs.to(torch.device(0))
            targets = [{k: v.cuda() for k, v in t.items()} for t in targets]
            
            model.train()
            set_criterion.train()
            
            outputs = model(imgs,queries)
            losses = set_criterion(outputs,targets)
            
            loss_to_optim = sum(
                losses[k] * wts_dict[k] for k in losses.keys() if k in wts_dict)
            
            optimizer.zero_grad()
            loss_to_optim.backward()
            if cfg.training.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    backbone_params + other_params, 
                    cfg.training.clip_max_norm)
            optimizer.step()

            if step%cfg.training.log_step==0:
                loss_str = f'Epoch: {epoch} | Iter: {it} | Step: {step} | '
                writer.add_scalar('Epoch',epoch,step)
                writer.add_scalar('Iter',it,step)
                writer.add_scalar('Step',step,step)
                writer.add_scalar(
                    'Lr/all_except_backbone',
                    lr_scheduler.get_last_lr()[0],
                    step)
                writer.add_scalar(
                    'Lr/backbone',
                    lr_scheduler.get_last_lr()[1],
                    step)
                writer.add_scalar(
                    'GradNorm/backbone',
                    grad_norm(backbone_params),
                    step)
                writer.add_scalar(
                    'GradNorm/other',
                    grad_norm(other_params),
                    step)
                writer.add_scalar(
                    'GradNorm/backbone+other',
                    grad_norm(backbone_params + other_params),
                    step)
                for k in itertools.chain(
                        wts_dict.keys(),['class_error','cardinality_error']):
                    loss_value = round(losses[k].item(),4)
                    loss_str += f'{k}: {loss_value} | '
                    writer.add_scalar(f'Loss/{k}/train',loss_value,step)
                print(loss_str)

            if step%cfg.training.vis_step==0:
                with torch.no_grad():
                    model.eval()
                    visualize(model,dataloaders['val'],cfg,step)

            if step%(10*cfg.training.log_step)==0:
                print(cfg.exp_name)
                    
            if step%cfg.training.ckpt_step==0:
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'iter': it,
                    'step': step,
                    'lr': lr_scheduler.get_last_lr()
                }, os.path.join(cfg.ckpt_dir,str(step).zfill(6)+'.pth'))

            step += 1

        lr_scheduler.step()
        
        if epoch%cfg.training.eval_epoch==0:
            with torch.no_grad():
                model.eval()
                AP = {}
                for subset, dataloader in dataloaders.items():
                    AP[subset] = eval_model(model,dataloaders[subset],cfg)[0]['AP']
                    writer.add_scalar(f'AP/{subset}',AP[subset],epoch)

                print('Epoch:',epoch,'AP',AP)

def visualize(model,dataloader,cfg,step):
    vis_dir = os.path.join(cfg.exp_dir,'train_vis/'+str(step).zfill(6))
    io.mkdir_if_not_exists(vis_dir,recursive=True)
    io.mkdir_if_not_exists(cfg.ckpt_dir,recursive=True)

    html_writer = HtmlWriter(os.path.join(vis_dir,'index.html'))
    html_writer.add_element({
        0: 'query',
        1: 'detection'})
    count = 0
    for data in dataloader:
        imgs, queries, targets = data
        imgs = imgs.to(torch.device(0))

        outputs = model(imgs,queries)
        imgs = dataloader.dataset.get_images_from_tensor(imgs)
        imgs = imgs.detach().cpu().numpy().astype(np.uint8)

        # visualize predictions
        pred_prob = outputs['pred_logits'].softmax(-1)
        topk = torch.topk(pred_prob[:,:,0],k=5,dim=1)
        topk_ids = topk.indices.detach().cpu().numpy()
        topk_values = topk.values.detach().cpu().numpy()
        pred_boxes = outputs['pred_boxes'].detach().cpu().numpy()
        gt_boxes = [t['boxes'].detach().numpy() for t in targets]
        B = pred_boxes.shape[0]
        for b in range(B):
            if count+b >= cfg.training.num_vis_samples:
                break

            # visualize prediction
            boxes = pred_boxes[b,topk_ids[b]]
            
            print(boxes)
            print(pred_prob[b,topk_ids[b]])
            vis_img = imgs[b]
            for k in range(gt_boxes[b].shape[0]):
                vis_bbox(boxes[k],vis_img,modify=True)

            # visualize gt
            boxes = gt_boxes[b]
            #vis_img = img[b]
            for k in range(boxes.shape[0]):
                vis_bbox(boxes[k],vis_img,color=(0,255,0),modify=True)

            fname = str(step).zfill(6) + '_' + str(count+b).zfill(4) + '.png'
            skio.imsave(os.path.join(vis_dir,fname),vis_img)

            html_writer.add_element({
                0: queries[b],
                1: html_writer.image_tag(fname),
                2: topk_values[b]})
        
        count += B
    
    html_writer.close()


@hydra.main(config_path=f'../../configs',config_name=f"exp/detr_lang")
def main(cfg):
    print(cfg.pretty())

    model = GPV(cfg.model).cuda()
    
    datasets = {
        'train': ClevrQueryDetectionTrainTask(cfg.task.clevr_query_detection_train,'train'),
        'val': ClevrQueryDetectionTrainTask(cfg.task.clevr_query_detection_train,'train')}
    
    dataloaders = {}
    for subset,dataset in datasets.items():
        dataloaders[subset] = dataset.get_dataloader(
            batch_size=cfg.training.batch_size,
            num_workers=cfg.training.num_workers)

    train_model(model,dataloaders,cfg)


if __name__=='__main__':
    main()

