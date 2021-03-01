import os
import nltk
import h5py
import hydra
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torchvision.models import resnet50
import itertools
import imagesize
import numpy as np
import skimage.io as skio
from utils.detr_misc import collate_fn as detr_collate_fn
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader
from tqdm import tqdm
from warmup_scheduler import GradualWarmupScheduler
from pytorch_transformers.optimization import WarmupLinearSchedule

from data.coco.synonyms import SYNONYMS
from exp.gpv_box_text import evaluators
from datasets.coco_multitask_dataset import CocoMultitaskDataset
import utils.io as io

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = resnet50(pretrained=True)
        self.cnn.fc = nn.Linear(2048,80)
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
        self.cls_to_idx = {l:i for i,l in enumerate(SYNONYMS.keys())}

    def get_targets(self,targets):
        tgts = [self.cls_to_idx[t['answer']] for t in targets]
        return torch.LongTensor(tgts)

    def forward(self,imgs,targets=None):
        logits = self.cnn(imgs)

        if targets is None:
            return logits

        tgts = self.get_targets(targets)
        tgts = tgts.cuda(imgs.device)
            
        return self.criterion(logits,tgts)


def cls_metric(model,dataloader,cfg):
    device = f'cuda:{cfg.gpu}'
    model.eval()
    correct = 0
    total = 0
    for data in tqdm(dataloader):
        if total >= cfg.training.num_val_samples['coco_cls']:
            break 

        imgs, queries, targets = data
        imgs = imgs.to(torch.device(device))
        for t in targets:
            for k,v in t.items():
                if not isinstance(v,str):
                    t[k] = v.cuda(device)
        
        imgs = imgs.tensors

        logits = model(imgs)
        tgts = model.get_targets(targets).cuda(imgs.device)
        preds = torch.argmax(logits,1)
        correct += torch.sum(tgts==preds).item()
        total += tgts.size(0)
    
    accuracy = round(correct / (total+1e-6),4)
    return accuracy


def get_lrs(optimizer):
    lrs = []
    for param_group in optimizer.param_groups:
        lrs.append(param_group['lr'])
    
    return lrs


def train_worker(gpu,cfg):
    cfg.gpu = gpu
    if cfg.gpu is not None:
        print("Use GPU: {} for training".format(cfg.gpu))

    model=Classifier()
    if cfg.gpu==0:
        print(model)

    datasets = {
        'train': CocoMultitaskDataset(cfg.learning_datasets,cfg.task_configs,'train'),
        'val': CocoMultitaskDataset(cfg.learning_datasets,cfg.task_configs,'val')
    }

    if cfg.multiprocessing_distributed:
        cfg.rank = cfg.rank * cfg.ngpus_per_node + cfg.gpu
        dist.init_process_group(
            backend=cfg.dist_backend, 
            init_method=cfg.dist_url,
            world_size=cfg.world_size,
            rank=cfg.rank)

        torch.cuda.set_device(cfg.gpu)
        model.cuda(cfg.gpu)
        get_targets = model.get_targets
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[cfg.gpu], find_unused_parameters=True)
        model.get_targets = get_targets

        # Create sampler for dataloader
        sampler = {'val': None}
        sampler['train'] = torch.utils.data.distributed.DistributedSampler(
            datasets['train'],shuffle=True)

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
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay)
    
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
        print(f'Loading checkpoint at epoch: {last_epoch}')

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
        if gpu==0:
            for eval_subset in ['train','val']:
                cls_acc = 0
                print(f'Evaluating on coco_cls {eval_subset}')
                eval_dataset = dataloaders[eval_subset].dataset.datasets['coco_cls']
                eval_dataloader = DataLoader(
                    eval_dataset,
                    batch_size=cfg.batch_size,
                    num_workers=cfg.workers,
                    shuffle=False,
                    collate_fn=detr_collate_fn)
                
                with torch.no_grad():
                    cls_acc = cls_metric(model,eval_dataloader,cfg)

                print(f'Dataset: coco_cls | Subset: {eval_subset} | Epoch: {epoch} | Acc: {cls_acc}')
                writer.add_scalar(f'cls_acc/{eval_subset}',cls_acc,step)
        
                if eval_subset=='val':
                    model_selection_metric = cls_acc

        if cfg.multiprocessing_distributed:
            sampler['train'].set_epoch(epoch)

        print(gpu,len(dataloaders['train']))
        for it,data in enumerate(dataloaders['train']):
            imgs, queries, targets = data
            imgs = imgs.to(torch.device(gpu))
            for t in targets:
                for k,v in t.items():
                    if not isinstance(v,str):
                        t[k] = v.cuda(device)

            model.train()
            imgs = imgs.tensors

            loss = model(imgs,targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if gpu==0 and step%cfg.training.log_step==0:
                loss_str = f'Epoch: {epoch} | Iter: {it} | Step: {step} | '
                if cfg.training.lr_linear_decay:
                    loss_str += f' LR: {warmup_scheduler.get_last_lr()[0]} | '

                loss_value = round(loss.item(),4)
                loss_str += f'Loss: {loss_value} | '
                writer.add_scalar(f'Loss/train',loss_value,step)
                writer.add_scalar('Epoch',epoch,step)
                writer.add_scalar('Iter',it,step)
                writer.add_scalar('Step',step,step)
                writer.add_scalar('Best Epoch',best_epoch,step)
                for j,group_lr in enumerate(get_lrs(optimizer)):
                    writer.add_scalar(
                        f'Lr/optimizer/group_{j}',
                        group_lr,
                        step)
                        
                print(loss_str)
            
            if gpu==0 and step%(10*cfg.training.log_step)==0:
                print('Exp:',cfg.exp_name)
                    
            if gpu==0 and step%cfg.training.ckpt_step==0:
                if model_selection_metric > best_metric:
                    print('Saving checkpoint ...')
                    best_metric = model_selection_metric
                    best_epoch = epoch
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'iter': it,
                        'step': step,
                        'lr': lr_scheduler.get_last_lr(),
                        'model_selection_metric': model_selection_metric,
                        'warmup_scheduler': warmup_scheduler.state_dict() if cfg.training.lr_linear_decay else None,
                    }, os.path.join(cfg.ckpt_dir,'model.pth'))

            step += 1
            launch=False

            if cfg.training.lr_linear_decay:
                warmup_scheduler.step()
            elif cfg.training.lr_warmup is True and epoch==0 and it < warmup_iters:
                warmup_scheduler.step(it)

        if not cfg.training.lr_linear_decay:
            lr_scheduler.step()


@hydra.main(config_path=f'../../configs',config_name=f"exp/cls_baseline")
def main(cfg):
    io.mkdir_if_not_exists(cfg.ckpt_dir,recursive=True)
    io.mkdir_if_not_exists(cfg.tb_dir,recursive=True)
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