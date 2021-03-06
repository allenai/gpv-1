import hydra
import numpy as np
from torch.utils.data import DataLoader, Dataset

from utils.detr_misc import collate_fn as detr_collate_fn
from .coco_datasets import *


class CocoMultitaskDataset(Dataset):
    def __init__(self,learning_datasets,task_configs,subset):
        super().__init__()
        self.datasets = {}
        self.sample_l = []
        self.sample_u = []
        for dataset_cls, info in learning_datasets.items():
            task_cfg = task_configs[info.task_config]
            self.datasets[info.name] = \
                globals()[dataset_cls](task_cfg,subset)
            L = len(self.datasets[info.name])
            if len(self.sample_l)==0:
                self.sample_l.append(0)
                self.sample_u.append(L)
            else:
                self.sample_l.append(self.sample_u[-1])
                self.sample_u.append(self.sample_u[-1]+L)

        self.sample_l = np.array(self.sample_l)
        self.sample_u = np.array(self.sample_u)
        self.dataset_names = list(self.datasets.keys())

    def __len__(self):
        N = 0
        for dataset_name, dataset in self.datasets.items():
            N += len(dataset)
        
        return N

    def __getitem__(self,i):
        cond = (i>=self.sample_l) * (i<self.sample_u)
        dataset_idx = cond.tolist().index(True)
        dataset_name = self.dataset_names[dataset_idx]
        return self.datasets[dataset_name][i-self.sample_l[dataset_idx]]

    def get_collate_fn(self):
        return detr_collate_fn
                
    def get_dataloader(self,**kwargs):
        collate_fn = self.get_collate_fn()
        return DataLoader(self,collate_fn=collate_fn,**kwargs)


@hydra.main(config_path=f'../configs',config_name=f"test/coco_datasets")
def main(cfg):
    tasks = CocoMultitaskDataset(cfg.learning_datasets,cfg.task_configs,'train')
    dataloader = tasks.get_dataloader(batch_size=10,shuffle=True)
    for data in dataloader:
        import ipdb; ipdb.set_trace()


if __name__=='__main__':
    main()