import hydra
import numpy as np
import skimage.io as skio
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate

import utils.io as io


class ClevrDetectionTrainTask(Dataset):
    def __init__(self,cfg,subset):
        self.cfg = cfg
        self.subset = subset

    def read_image(self,i):
        img_dir = self.cfg[self.subset].images
        img_path = f'{img_dir}/CLEVR_new_{str(i).zfill(6)}.png'
        return skio.imread(img_path)[:,:,:3]
    
    def read_scene(self,i):
        scene_dir = self.cfg[self.subset].scenes
        scene_path = f'{scene_dir}/CLEVR_new_{str(i).zfill(6)}.json'
        return io.load_json_object(scene_path)

    def get_bboxes(self,scene,fmt='xyxy'):
        bboxes = []

        for object_info in scene['objects']:
            if fmt=='xyxy':
                bbox = [
                    object_info['x'],
                    object_info['y'],
                    object_info['x'] + object_info['width'],
                    object_info['y'] + object_info['height']]
            elif fmt=='cxcywh':
                bbox = [
                    object_info['x'] + 0.5*object_info['width'],
                    object_info['y'] + 0.5*object_info['height'],
                    object_info['width'],
                    object_info['height']]
            
            bboxes.append(bbox)
        
        return np.array(bboxes).astype(np.float32)

    def __len__(self):
        return self.cfg[self.subset].num_images
    
    def __getitem__(self,i):
        # Input
        img = self.read_image(i).astype(np.float32)
        norm_img = (img/255) - 0.5
        inputs = {
            'img': img,
            'norm_img': norm_img
        }

        # Output
        scene = self.read_scene(i)
        bboxes = self.get_bboxes(scene)
        outputs = {
            'bboxes': bboxes
        }
    
        return {**inputs, **outputs}

    def get_collate_fn(self):
        def collate_fn(batch):
            batch = [sample for sample in batch if sample is not None]
            if len(batch)==0:
                return None

            collated_batch = {}
            for k in batch[0].keys():
                collated_batch[k] = [sample[k] for sample in batch]
                if k not in ['bboxes']:
                    collated_batch[k] = default_collate(collated_batch[k])
            
            return collated_batch
        
        return collate_fn
                
    def get_dataloader(self,**kwargs):
        collate_fn = self.get_collate_fn()
        return DataLoader(self,collate_fn=collate_fn,**kwargs)


@hydra.main(config_name="../configs/clevr_detection_train_task.yaml")
def test_dataset(cfg):
    dataset = ClevrDetectionTrainTask(cfg,'train')
    dataloader = dataset.get_dataloader(batch_size=2)
    for data in dataloader:
        import ipdb; ipdb.set_trace()

if __name__=='__main__':
    test_dataset()
