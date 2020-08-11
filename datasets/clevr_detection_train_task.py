import hydra
import numpy as np
import skimage.io as skio
from skimage.transform import rescale
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate

import utils.io as io
from utils.detr_misc import collate_fn


class ClevrDetectionTrainTask(Dataset):
    def __init__(self,cfg,subset):
        self.cfg = cfg
        self.subset = subset

    def read_image(self,i):
        img_dir = self.cfg[self.subset].images
        img_path = f'{img_dir}/CLEVR_new_{str(i).zfill(6)}.png'
        img = skio.imread(img_path)[:,:,:3]
        return rescale(img,(self.cfg.scale,self.cfg.scale,1),anti_aliasing=True)
    
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
        
        return self.cfg.scale*np.array(bboxes).astype(np.float32)

    def normalize_bbox(self,bbox,H,W):
        bbox = np.copy(bbox)
        bbox[:,0] = bbox[:,0] / W   # x1 or cx
        bbox[:,1] = bbox[:,1] / H   # y1 or cy
        bbox[:,2] = bbox[:,2] / W   # x2 or w
        bbox[:,3] = bbox[:,3] / H   # y2 or h
        return bbox

    def __len__(self):
        return self.cfg[self.subset].num_images
    
    def __getitem__(self,i):
        # Input
        img = self.read_image(i).astype(np.float32)
        img = (img - 0.5)/0.25
        img = torch.as_tensor(img,dtype=torch.float32).permute(2,0,1)
        _,H,W = img.size()

        # Output
        scene = self.read_scene(i)
        bboxes_cxcywh = self.get_bboxes(scene,'cxcywh')
        bboxes_ncxcywh = self.normalize_bbox(bboxes_cxcywh,H,W)
        labels = np.zeros([bboxes_cxcywh.shape[0]])
        targets = {
            'labels': torch.as_tensor(labels,dtype=torch.long),
            'boxes': torch.as_tensor(bboxes_ncxcywh,dtype=torch.float32)
        }
    
        return img, targets

    def get_collate_fn(self):
        return collate_fn
                
    def get_dataloader(self,**kwargs):
        collate_fn = self.get_collate_fn()
        return DataLoader(self,collate_fn=collate_fn,**kwargs)


@hydra.main(config_path="../configs",config_name="test/clevr_detection_train_task_dataset")
def test_dataset(cfg):
    print(cfg.pretty())
    dataset = ClevrDetectionTrainTask(cfg.task.clevr_detection_train,'train')
    dataloader = dataset.get_dataloader(batch_size=2)
    for data in dataloader:
        import ipdb; ipdb.set_trace()

if __name__=='__main__':
    test_dataset()
