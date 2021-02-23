import os
import nltk
import h5py
import hydra
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
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

from exp.gpv_box_text import evaluators
from datasets.coco_multitask_dataset import CocoMultitaskDataset
import utils.io as io

def train_worker(gpu,cfg):
    cfg.gpu = gpu
    if cfg.gpu is not None:
        print("Use GPU: {} for training".format(cfg.gpu))

    model=resnet50()
    model.fc = nn.Linear(2048,80)
    model.cuda(cfg.gpu)

    datasets = {
        'train': CocoMultitaskDataset(cfg.learning_datasets,cfg.task_configs,'train'),
        'val': CocoMultitaskDataset(cfg.learning_datasets,cfg.task_configs,'val')
    }

    dataloaders = {}
    for subset,dataset in datasets.items():
        dataloaders[subset] = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            collate_fn=detr_collate_fn,
            num_workers=cfg.workers,
            pin_memory=True,
            shuffle=(subset=='train'))

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
            import ipdb; ipdb.set_trace()
            y = model(imgs)
            import ipdb; ipdb.set_trace()

@hydra.main(config_path=f'../../configs',config_name=f"exp/cls_baseline")
def main(cfg):
    io.mkdir_if_not_exists(cfg.ckpt_dir,recursive=True)
    io.mkdir_if_not_exists(cfg.tb_dir,recursive=True)
    if cfg.training.freeze:
        cfg.training.batch_size = cfg.training.frozen_batch_size
        cfg.batch_size = cfg.training.frozen_batch_size
    train_worker(cfg.gpu,cfg)

if __name__=='__main__':
    main()