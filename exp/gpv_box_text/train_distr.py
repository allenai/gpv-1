import os
import hydra
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
import itertools
import numpy as np
import skimage.io as skio

from .models.gpv import GPV
from .models.losses import GPVCriterion
from .eval import eval_model
from datasets.coco_multitask_dataset import CocoMultitaskDataset
from utils.bbox_utils import vis_bbox
from utils.detr_misc import collate_fn
import utils.io as io
from utils.html_writer import HtmlWriter


def grad_norm(params):
    total_norm = 0
    for p in params:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    
    return total_norm ** (1. / 2)
    

def visualize(model,dataloader,cfg,step,subset):
    device = f'cuda:{cfg.gpu}'
    vis_dir = os.path.join(
        cfg.exp_dir,
        f'training_visualizations/{subset}_'+str(step).zfill(6))
    io.mkdir_if_not_exists(vis_dir,recursive=True)
    io.mkdir_if_not_exists(cfg.ckpt_dir,recursive=True)
    word_to_idx = model.word_to_idx
    idx_to_word = [None]*len(word_to_idx)
    for word,idx in word_to_idx.items():
        idx_to_word[idx] = word

    html_writer = HtmlWriter(os.path.join(vis_dir,'index.html'))
    html_writer.add_element({
        0: 'query',
        1: 'detection',
        2: 'pred_answer',
        3: 'gt_answer',
        4: 'relevance_scores'})
    count = 0
    finish_vis = False
    for data in dataloader:
        imgs, queries, targets = data
        imgs = imgs.to(torch.device(device))
        for t in targets:
            for k,v in t.items():
                if k!='answer':
                    t[k] = v.cuda(device)
        
        answer_tokens,answer_token_ids = model.encode_answers(targets)
        for i,t in enumerate(targets):
            t['answer_token_ids'] = answer_token_ids[i,1:]

        outputs = model(imgs,queries,answer_token_ids=None)
        dataset_name = list(dataloader.dataset.datasets.keys())[0]
        imgs = dataloader.dataset.datasets[dataset_name].get_images_from_tensor(imgs)
        imgs = imgs.detach().cpu().numpy().astype(np.uint8)

        # visualize predictions
        pred_prob = outputs['pred_relevance_logits'].softmax(-1)
        topk = torch.topk(pred_prob[:,:,0],k=5,dim=1)
        topk_ids = topk.indices.detach().cpu().numpy()
        topk_values = topk.values.detach().cpu().numpy()
        pred_boxes = outputs['pred_boxes'].detach().cpu().numpy()
        gt_boxes = [None]*len(targets)
        for i,t in enumerate(targets):
            if 'boxes' in t:
                gt_boxes[i] = t['boxes'].detach().cpu().numpy()
        #gt_boxes = [t['boxes'].detach().cpu().numpy() for t in targets]

        topk_answers = torch.topk(outputs['answer_logits'][-1],k=1,dim=-1)
        topk_answer_ids = topk_answers.indices.detach().cpu().numpy()
        pred_answers = model.token_ids_to_words(topk_answer_ids[:,:,0])
        B = pred_boxes.shape[0]
        for b in range(B):
            if count+b >= cfg.training.num_vis_samples:
                finish_vis = True
                break

            # visualize prediction
            boxes = pred_boxes[b,topk_ids[b]]
            if gt_boxes[b] is None:
                num_gt_boxes = 0
            else:
                num_gt_boxes = gt_boxes[b].shape[0]
            vis_img = imgs[b]            
            for k in range(num_gt_boxes,max(num_gt_boxes,5)):
                vis_bbox(boxes[k],vis_img,color=(0,0,255),modify=True,alpha=0)
            
            for k in range(min(num_gt_boxes,5)):
                vis_bbox(boxes[k],vis_img,color=(255,0,0),modify=True,alpha=0)

            # visualize gt
            boxes = gt_boxes[b]
            if boxes is not None:
                for k in range(boxes.shape[0]):
                    vis_bbox(boxes[k],vis_img,color=(0,255,0),modify=True,alpha=0)

            fname = str(step).zfill(6) + '_' + str(count+b).zfill(4) + '.png'
            skio.imsave(os.path.join(vis_dir,fname),vis_img)

            html_writer.add_element({
                0: queries[b],
                1: html_writer.image_tag(fname),
                2: pred_answers[b],
                3: answer_tokens[b],
                4: np.round(topk_values[b],4)})
        
        if finish_vis is True:
            break
        
        count += B
    
    html_writer.close()


def freeze_detr_params(model):
    print(f'Setting requires grad to False for DETR params')
    for n,p in model.named_parameters():
        if n in model.init_detr_params:
            p.requires_grad = False
            #print(f'    {n}')


def train_worker(gpu,cfg):
    cfg.gpu = gpu
    if cfg.gpu is not None:
        print("Use GPU: {} for training".format(cfg.gpu))

    if gpu==0:
        print(cfg.pretty())

    model = GPV(cfg.model)
    model.load_pretr_detr()
    if cfg.training.freeze is True:
        freeze_detr_params(model)

    datasets = {
        'train': CocoMultitaskDataset(cfg.learning_datasets,cfg.task_configs,'train'),
        'val': CocoMultitaskDataset(cfg.learning_datasets,cfg.task_configs,'val')
    }
    for subset,dataset in datasets.items():
        print(f'{subset} set size:',len(dataset))

    if cfg.multiprocessing_distributed:
        cfg.rank = cfg.rank * cfg.ngpus_per_node + cfg.gpu
        dist.init_process_group(
            backend=cfg.dist_backend, 
            init_method=cfg.dist_url,
            world_size=cfg.world_size,
            rank=cfg.rank)

        # Setup distributed model
        torch.cuda.set_device(cfg.gpu)
        model.cuda(cfg.gpu)
        init_detr_params = model.init_detr_params
        word_to_idx = model.word_to_idx
        encode_answers = model.encode_answers
        token_ids_to_words = model.token_ids_to_words
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[cfg.gpu], find_unused_parameters=True)
        model.encode_answers = encode_answers
        model.word_to_idx = word_to_idx
        model.token_ids_to_words = token_ids_to_words
        model.init_detr_params = init_detr_params

        # Create sampler for dataloader
        sampler = {'val': None}
        sampler['train'] = torch.utils.data.distributed.DistributedSampler(
            datasets['train'],shuffle=True)
    else:
        model.cuda(cfg.gpu)
        sampler = {'train': None, 'val': None}

    dataloaders = {}
    for subset,dataset in datasets.items():
        dataloaders[subset] = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            collate_fn=collate_fn,
            num_workers=cfg.workers,
            pin_memory=True,
            shuffle=(sampler[subset] is None and subset is 'train'),
            sampler=sampler[subset])

    device = f'cuda:{cfg.gpu}'
    if gpu==0:
        writer = SummaryWriter(log_dir=cfg.tb_dir)

    backbone_params = [p for n, p in model.named_parameters() \
            if 'backbone' in n and p.requires_grad]
    other_params = [p for n, p in model.named_parameters() \
            if ('backbone' not in n) and ('bert.' not in n) and p.requires_grad]
    param_dicts = [
        {'params': other_params},
        {'params': backbone_params, 'lr': cfg.model.detr.lr_backbone}]
    optimizer = torch.optim.AdamW(
        param_dicts, 
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay)

    gpv_criterion = GPVCriterion(cfg.losses).cuda(device)

    step = 0
    last_epoch = -1
    if cfg.training.ckpt is not None:
        ckpt = torch.load(cfg.training.ckpt)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        step = ckpt['step']
        last_epoch = ckpt['epoch']

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        cfg.training.lr_drop,
        last_epoch=last_epoch)

    for epoch in range(last_epoch+1,cfg.training.num_epochs):
        if cfg.multiprocessing_distributed:
            sampler['train'].set_epoch(epoch)

        print(gpu,len(dataloaders['train']))
        for it,data in enumerate(dataloaders['train']):
            imgs, queries, targets = data
            imgs = imgs.to(torch.device(gpu))
            for t in targets:
                for k,v in t.items():
                    if k!='answer':
                        t[k] = v.cuda(device)
            
            model.train()
            gpv_criterion.train()

            answer_tokens,answer_token_ids = model.encode_answers(targets)
            for i,t in enumerate(targets):
                t['answer_token_ids'] = answer_token_ids[i,1:]
            
            outputs = model(imgs,queries,answer_token_ids,targets)
            total_loss = outputs
            losses = {
                'total_loss': total_loss
            }
            #total_loss, losses = gpv_criterion(outputs,targets)
            if total_loss is not None:
                optimizer.zero_grad()
                total_loss.backward()
                if cfg.training.clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        backbone_params + other_params, 
                        cfg.training.clip_max_norm)
                optimizer.step()
            
            if gpu==0 and step%cfg.training.log_step==0:
                loss_str = f'Epoch: {epoch} | Iter: {it} | Step: {step} | '
                loss_value = round(total_loss.item(),4)
                loss_str += f'total_loss: {loss_value} | '
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
                for loss_name,loss_value in losses.items():
                    if loss_value is None:
                        continue
                    loss_value = round(loss_value.item(),4)
                    loss_str += f'{loss_name}: {loss_value} | '
                    writer.add_scalar(f'Loss/{loss_name}/train',loss_value,step)
                print(loss_str)

            if gpu==0 and step%cfg.training.vis_step==0:
                with torch.no_grad():
                    model.eval()
                    for subset in ['train','val']:
                        print(f'Visualizing {subset} ...')
                        visualize(model,dataloaders[subset],cfg,step,subset)

            if gpu==0 and step%(10*cfg.training.log_step)==0:
                print('Exp:',cfg.exp_name)
                    
            if gpu==0 and step%cfg.training.ckpt_step==0:
                print('Saving checkpoint ...')
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'iter': it,
                    'step': step,
                    'lr': lr_scheduler.get_last_lr()
                }, os.path.join(cfg.ckpt_dir,'model.pth'))

            step += 1

        lr_scheduler.step()

@hydra.main(config_path=f'../../configs',config_name=f"exp/gpv_box_text_coco")
def main(cfg):
    io.mkdir_if_not_exists(cfg.ckpt_dir,recursive=True)
    io.mkdir_if_not_exists(cfg.tb_dir,recursive=True)

    if cfg.training.freeze:
        cfg.training.num_epochs = cfg.training.frozen_epochs
        cfg.training.batch_size = cfg.training.frozen_batch_size
        cfg.batch_size = cfg.training.frozen_batch_size

    if cfg.multiprocessing_distributed:
        cfg.world_size = cfg.ngpus_per_node * cfg.num_nodes
        cfg.batch_size = int(cfg.batch_size / cfg.ngpus_per_node)
        cfg.workers = int(
            (cfg.workers + cfg.ngpus_per_node - 1) / cfg.ngpus_per_node)
        mp.spawn(train_worker, nprocs=cfg.ngpus_per_node, args=(cfg,))
    else:
        train_worker(cfg.gpu,cfg)
    
    


if __name__=='__main__':
    main()

