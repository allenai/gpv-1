import os
import nltk
import h5py
import hydra
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
import itertools
import imagesize
import numpy as np
import skimage.io as skio
from utils.detr_misc import collate_fn as detr_collate_fn
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader
from nltk.tokenize.treebank import TreebankWordDetokenizer
from tqdm import tqdm
from warmup_scheduler import GradualWarmupScheduler
from pytorch_transformers.optimization import WarmupLinearSchedule

from .models.gpv import GPV
from .models.losses import GPVCriterion
from .metrics import *
from datasets.coco_multitask_dataset import CocoMultitaskDataset
from utils.bbox_utils import vis_bbox
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
                if not isinstance(v,str):
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


def freeze_detr_params(model,requires_grad=False):
    print(f'Setting requires grad to False for DETR params')
    for n,p in model.named_parameters():
        if n in model.init_detr_params:
            p.requires_grad = requires_grad


def get_lrs(optimizer):
    lrs = []
    for param_group in optimizer.param_groups:
        lrs.append(param_group['lr'])
    
    return lrs

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

        torch.cuda.set_device(cfg.gpu)
        
        dist.init_process_group(
            backend=cfg.dist_backend, 
            init_method=cfg.dist_url,
            world_size=cfg.world_size,
            rank=cfg.rank)

        model = GPV(cfg.model)
        model.load_pretr_detr()
        if cfg.training.freeze is True:
            freeze_detr_params(model)

        model.cuda(cfg.gpu)
        init_detr_params = model.init_detr_params
        word_to_idx = model.word_to_idx
        encode_answers = model.encode_answers
        token_ids_to_words = model.token_ids_to_words
        vocab = model.vocab
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[cfg.gpu], find_unused_parameters=True)
        model.encode_answers = encode_answers
        model.word_to_idx = word_to_idx
        model.token_ids_to_words = token_ids_to_words
        model.init_detr_params = init_detr_params
        model.vocab = vocab

        # Create sampler for dataloader
        sampler = {'val': None}
        sampler['train'] = torch.utils.data.distributed.DistributedSampler(
            datasets['train'],shuffle=True)
    else:
        model = GPV(cfg.model)
        model.load_pretr_detr()
        if cfg.training.freeze is True:
            freeze_detr_params(model)

        model.cuda(cfg.gpu)
        sampler = {'train': None, 'val': None}

    dataloaders = {}
    for subset,dataset in datasets.items():
        dataloaders[subset] = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            collate_fn=detr_collate_fn,
            num_workers=cfg.workers,
            pin_memory=True,
            shuffle=(sampler[subset] is None), #(sampler[subset] is None and subset is 'train'),
            sampler=sampler[subset])

    device = f'cuda:{cfg.gpu}'
    if gpu==0:
        writer = SummaryWriter(log_dir=cfg.tb_dir)

    params = {
        'detr_backbone': [],
        'detr_head': [],
        'bert': [],
        'others': []
    }
    for n,p in model.named_parameters():
        if 'detr.backbone' in n:
            params['detr_backbone'].append(p)
        elif 'detr' in n:
            params['detr_head'].append(p)
        elif 'bert.' in n:
            params['bert'].append(p)
        else:
            params['others'].append(p)

    for k,v in params.items(): 
        print(k,len(v))
    
    optimizer = torch.optim.AdamW([
        {'params': params['detr_backbone'], 'lr': cfg.training.lr_backbone},
        {'params': params['detr_head']},
        {'params': params['bert']},
        {'params': params['others']}],
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay)

    gpv_criterion = GPVCriterion(cfg.losses).cuda(device)

    step = 0
    last_epoch = -1
    model_selection_metric = 0
    best_metric = 0
    best_epoch = -1
    if cfg.training.ckpt is not None:
        loc = 'cuda:{}'.format(cfg.gpu)
        ckpt = torch.load(cfg.training.ckpt, map_location=loc)
        state_dict = model.state_dict()
        for k,v in ckpt['model'].items():
            if k in state_dict and state_dict[k].size()==v.size():
                v.requires_grad = state_dict[k].requires_grad
                state_dict[k] = v
                print(k)

        model.load_state_dict(state_dict)
        optimizer.load_state_dict(ckpt['optimizer'])

        step = ckpt['step']
        last_epoch = ckpt['epoch']
        if model_selection_metric in ckpt:
            model_selection_metric = ckpt['model_selection_metric']
        else:
            model_selection_metric = 0
            
        # since a checkpoint is saved only if it is has the best metric so far
        best_metric = model_selection_metric
        best_epoch = last_epoch
        print(f'Loading checkpoint at the end of epoch {last_epoch}')

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        cfg.training.lr_milestones,
        cfg.training.lr_drop,
        last_epoch=last_epoch)
    
    warmup_iters = len(dataloaders['train'])
    if cfg.training.lr_warmup is True:
        if cfg.training.lr_linear_decay:
            num_train_optimization_steps = len(dataloaders['train']) * cfg.training.num_epochs
            warmup_steps = cfg.training.lr_warmup_fraction * num_train_optimization_steps
            warmup_scheduler = WarmupLinearSchedule(
                optimizer,
                warmup_steps=warmup_steps,
                t_total=num_train_optimization_steps,
                last_epoch=step)
        else:
            warmup_scheduler = GradualWarmupScheduler(
                optimizer,
                multiplier=1,
                total_epoch=warmup_iters,
                last_epoch=step) # updated every iter not epoch
            if gpu==0:
                print('Warmup iters:',warmup_iters)

        if cfg.training.ckpt is not None:
            warmup_scheduler.load_state_dict(ckpt['warmup_scheduler'])

    if cfg.training.lr_warmup and not cfg.training.lr_linear_decay:
        # zero grad step needed for warmup scheduler
        optimizer.zero_grad()
        optimizer.step()

    
    training_epochs = cfg.training.num_epochs
    if cfg.training.freeze is True:
        training_epochs = cfg.training.frozen_epochs

    launch = True
    for epoch in range(last_epoch+1,training_epochs):
        if gpu==0 and ((not launch) or cfg.training.run_eval_at_launch):
            for eval_subset in ['train','val']:
                vqa_acc = 0
                cls_acc = 0
                cider = 0
                det_map = 0
                for dataset_name in dataloaders[eval_subset].dataset.datasets:
                    print(f'Evaluating on {dataset_name} {eval_subset}')
                    eval_dataset = dataloaders[eval_subset].dataset.datasets[dataset_name]
                    eval_dataloader = DataLoader(
                        eval_dataset,
                        batch_size=cfg.batch_size,
                        num_workers=cfg.workers,
                        shuffle=False,
                        collate_fn=detr_collate_fn)
                    
                    if dataset_name=='coco_vqa':
                        with torch.no_grad():
                            vqa_acc = vqa_accuracy(model,eval_dataloader,cfg)

                        print(f'Dataset: {dataset_name} | Subset: {eval_subset} | Epoch: {epoch} | Acc: {vqa_acc}')
                        writer.add_scalar(f'vqa_acc/{eval_subset}',vqa_acc,step)

                    elif dataset_name=='coco_cls':
                        with torch.no_grad():
                            cls_acc = cls_metrics(model,eval_dataloader,cfg)

                        print(f'Dataset: {dataset_name} | Subset: {eval_subset} | Epoch: {epoch} | Acc: {cls_acc}')
                        writer.add_scalar(f'cls_acc/{eval_subset}',cls_acc,step)
                    
                    elif dataset_name=='coco_cap':
                        with torch.no_grad():
                            metrics = cap_metrics(model,eval_dataloader,cfg)
                            cider = metrics['Cider']
                            bleu1 =  metrics['Bleu1']
                            bleu4 =  metrics['Bleu4']

                        print(f'Dataset: {dataset_name} | Subset: {eval_subset} | Epoch: {epoch} | Bleu1: {bleu1} | Bleu4: {bleu4} | Cider: {cider}')
                        writer.add_scalar(f'cap_metrics/{eval_subset}/cider',cider,step)
                        writer.add_scalar(f'cap_metrics/{eval_subset}/bleu1',bleu1,step)
                        writer.add_scalar(f'cap_metrics/{eval_subset}/bleu4',bleu4,step)
                    
                    elif dataset_name=='coco_det':
                        with torch.no_grad():
                            det_map = det_metrics(model,eval_dataloader,cfg)

                        print(f'Dataset: {dataset_name} | Subset: {eval_subset} | Epoch: {epoch} | mAP: {det_map}')
                        writer.add_scalar(f'det_map/{eval_subset}',det_map,step)
                    
                    else:
                        print(f'Eval not implemented for {dataset_name}')
                
                if eval_subset=='val':
                    model_selection_metric = vqa_acc+cider+det_map+cls_acc

                    if model_selection_metric > best_metric:
                        print('Saving checkpoint ...')
                        best_metric = model_selection_metric
                        best_epoch = epoch-1
                        torch.save({
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'epoch': best_epoch,
                            'step': step,
                            'lr': lr_scheduler.get_last_lr(),
                            'model_selection_metric': model_selection_metric,
                            'warmup_scheduler': warmup_scheduler.state_dict() if cfg.training.lr_linear_decay else None,
                        }, os.path.join(cfg.ckpt_dir,'model.pth'))

        if cfg.multiprocessing_distributed:
            sampler['train'].set_epoch(epoch)

        for it,data in enumerate(dataloaders['train']):
            imgs, queries, targets = data
            imgs = imgs.to(torch.device(gpu))
            for t in targets:
                for k,v in t.items():
                    if not isinstance(v,str):
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
                        params['detr_backbone']+params['detr_head'], 
                        cfg.training.clip_max_norm)

                optimizer.step()
            
            if gpu==0 and step%cfg.training.log_step==0:
                loss_str = f'Epoch: {epoch} | Iter: {it} | Step: {step} | '
                if cfg.training.lr_linear_decay:
                    loss_str += f' LR: {warmup_scheduler.get_last_lr()[0]} | '
                loss_value = round(total_loss.item(),4)
                #loss_str += f'total_loss: {loss_value} | '
                writer.add_scalar('Epoch',epoch,step)
                writer.add_scalar('Iter',it,step)
                writer.add_scalar('Step',step,step)
                writer.add_scalar('Best Epoch',best_epoch,step)
                for j,group_lr in enumerate(get_lrs(optimizer)):
                    writer.add_scalar(
                        f'Lr/optimizer/group_{j}',
                        group_lr,
                        step)
                
                for loss_name,loss_value in losses.items():
                    if loss_value is None:
                        continue
                    loss_value = round(loss_value.item(),4)
                    loss_str += f'{loss_name}: {loss_value} | '
                    writer.add_scalar(f'Loss/{loss_name}/train',loss_value,step)
                print(loss_str)

            if gpu==0 and step%cfg.training.vis_step==0 and \
                ((not launch) or cfg.training.run_vis_at_launch):
                with torch.no_grad():
                    model.eval()
                    for subset in ['train','val']:
                        print(f'Visualizing {subset} ...')
                        visualize(model,dataloaders[subset],cfg,step,subset)

            if gpu==0 and step%(10*cfg.training.log_step)==0:
                print('Exp:',cfg.exp_name)
                
            step += 1
            launch=False

            if cfg.training.lr_linear_decay:
                warmup_scheduler.step()
            elif cfg.training.lr_warmup is True and epoch==0 and it < warmup_iters:
                warmup_scheduler.step(it)

        if not cfg.training.lr_linear_decay:
            lr_scheduler.step()

        

@hydra.main(config_path=f'../../configs',config_name=f"exp/gpv")
def main(cfg):
    io.mkdir_if_not_exists(cfg.ckpt_dir,recursive=True)
    io.mkdir_if_not_exists(cfg.tb_dir,recursive=True)
    nltk.download('punkt')
    
    if cfg.training.freeze:
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

