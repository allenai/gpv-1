import os
import hydra
import numpy as np
import skimage.io as skio
from skimage.transform import resize
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
import torchvision.transforms as T

import utils.io as io
from utils.detr_misc import collate_fn as detr_collate_fn


class GenericCocoDataset(Dataset):
    def __init__(self,cfg,subset):
        super().__init__()
        self.cfg = cfg
        self.subset = subset
        self.samples = io.load_json_object(self.cfg.samples[self.subset])
        self.imh = self.cfg.image_size.H
        self.imw = self.cfg.image_size.W
        self.mean = torch.FloatTensor([0.485, 0.456, 0.406])
        self.std = torch.FloatTensor([0.229, 0.224, 0.225])
        if subset=='train':
            self.transforms = T.Compose([
                T.ToPILImage(mode='RGB'),
                T.RandomApply([
                    T.ColorJitter(0.2, 0.2, 0.2, 0.0)
                ], p=0.8),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        if subset in ['val','test']:
            self.transforms = T.Compose([
                T.ToPILImage(mode='RGB'),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        max_samples = self.cfg.max_samples[self.subset]
        num_samples = len(self.samples)
        if max_samples is None:
            return num_samples
        else:
            return min(num_samples,max_samples)

    def read_image(self,image_subset,image_id):
        img_dir = os.path.join(self.cfg.image_dir,image_subset)
        img_path = os.path.join(
            img_dir,
            'COCO_'+image_subset+'_'+str(image_id).zfill(12)+'.jpg')
        img = skio.imread(img_path)
        if len(img.shape)==2:
            img = np.tile(np.expand_dims(img,2),(1,1,3))
        else:
            img = img[:,:,:3]
        
        original_image_size = img.shape[:2] # HxW
        resized_image_size = resize(img,(self.imh,self.imw),anti_aliasing=True)
        return resized_image_size, original_image_size

    def get_boxes(self,coco_boxes,fmt='xyxy'):
        boxes = []
        for coco_box in coco_boxes:
            x,y,w,h = coco_box
            if fmt=='xyxy':
                box = [x,y,x+w,y+h]
            elif fmt=='cxcywh':
                box = [x+0.5*w,y+0.5*h,w,h]
            
            boxes.append(box)
        
        return np.array(boxes).astype(np.float32)

    def normalize_bbox(self,bbox,H,W):
        bbox = np.copy(bbox)
        bbox[:,0] = bbox[:,0] / W   # x1 or cx
        bbox[:,1] = bbox[:,1] / H   # y1 or cy
        bbox[:,2] = bbox[:,2] / W   # x2 or w
        bbox[:,3] = bbox[:,3] / H   # y2 or h
        return bbox

    def __getitem__(self,i):
        sample = self.samples[i]

        if self.cfg.read_image is True:
            image_subset = sample['image']['subset']
            image_id = sample['image']['image_id']
            img, original_image_size = self.read_image(
                image_subset,image_id)
            #img = img.astype(np.float32)
            #img = (img - 0.5)/0.25
            #img = torch.as_tensor(img,dtype=torch.float32).permute(2,0,1)
            #import ipdb; ipdb.set_trace()
            img = (255*img).astype(np.uint8)
            img = self.transforms(img)
        # else:
        #     img = torch.zeros([3,self.imh,self.imw])
        
        query = sample['query']

        targets = {}
        if 'boxes' in sample:
            bboxes_cxcywh = self.get_boxes(sample['boxes'],'cxcywh')
            if self.cfg.read_image is not True:
                bboxes_ncxcywh = bboxes_cxcywh
            else:
                bboxes_ncxcywh = self.normalize_bbox(
                    bboxes_cxcywh,*original_image_size)

            labels = np.zeros([bboxes_cxcywh.shape[0]])
            targets.update({
                'labels': torch.as_tensor(labels,dtype=torch.long),
                'boxes': torch.as_tensor(bboxes_ncxcywh,dtype=torch.float32)
            })
        
        if 'answer' in sample:
            targets['answer'] = sample['answer']

        if self.cfg.read_image is True:
            return img, query, targets
        else:
            return query,targets
    
    def get_images_from_tensor(self,imgs):
        """
        imgs: Bx3xHxW torch.float32 normalized image tensor
        """
        device = imgs.tensors.device
        imgs = 255*(self.mean.cuda(device) + \
            self.std.cuda(device)*imgs.tensors.permute(0,2,3,1))
        #imgs = 255*(0.5+0.25*imgs.tensors.permute(0,2,3,1))
        return imgs

    def get_collate_fn(self):
        return detr_collate_fn
                
    def get_dataloader(self,**kwargs):
        collate_fn = self.get_collate_fn()
        return DataLoader(self,collate_fn=collate_fn,**kwargs)


@hydra.main(config_path="../configs",config_name="test/coco_datasets")
def test_dataset(cfg):
    print(cfg.pretty())
    dataset = GenericCocoDataset(cfg.task_configs.coco_detection,'train')
    dataloader = dataset.get_dataloader(batch_size=2)
    for data in dataloader:
        import ipdb; ipdb.set_trace()

if __name__=='__main__':
    test_dataset()