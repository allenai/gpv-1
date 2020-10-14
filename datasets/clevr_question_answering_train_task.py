import os
import hydra
import numpy as np
import skimage.io as skio
from skimage.transform import rescale
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate

import utils.io as io
from utils.detr_misc import collate_fn as detr_collate_fn


class ClevrQuestionAnsweringTrainTask(Dataset):
    def __init__(self,cfg,subset):
        self.cfg = cfg
        self.subset = subset
        self.samples = io.load_json_object(self.cfg[self.subset].samples)
        # self.vocab=[
        #     'small',
        #     'large',
        #     'rubber',
        #     'metal'
        # ]
        # self.word_to_idx = {w:i for i,w in enumerate(self.vocab)}

    def read_image(self,img_name):
        img_dir = self.cfg[self.subset].images
        img_path = os.path.join(img_dir,img_name)
        img = skio.imread(img_path)[:,:,:3]
        return rescale(img,(self.cfg.scale,self.cfg.scale,1),anti_aliasing=True)

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
        max_samples = self.cfg[self.subset].max_samples
        num_samples = len(self.samples)
        if max_samples is None:
            return num_samples
        else:
            return min(num_samples,max_samples)
    
    def __getitem__(self,i):
        sample = self.samples[i]

        # Input
        img = self.read_image(sample['image']).astype(np.float32)
        img = (img - 0.5)/0.25
        img = torch.as_tensor(img,dtype=torch.float32).permute(2,0,1)
        _,H,W = img.size()

        # Output
        bboxes_cxcywh = self.get_bboxes(sample,'cxcywh')
        bboxes_ncxcywh = self.normalize_bbox(bboxes_cxcywh,H,W)
        labels = np.zeros([bboxes_cxcywh.shape[0]])
        targets = {
            'labels': torch.as_tensor(labels,dtype=torch.long),
            #'boxes': torch.as_tensor(bboxes_ncxcywh,dtype=torch.float32),
            #'answer': torch.as_tensor(self.word_to_idx[sample['answer']],dtype=torch.long)
            'answer': sample['answer']
        }
        query = sample['query']
    
        return img, query, targets

    def get_images_from_tensor(self,imgs):
        """
        imgs: Bx3xHxW torch.float32 normalized image tensor
        """
        imgs = 255*(0.5+0.25*imgs.tensors.permute(0,2,3,1))
        return imgs

    def get_collate_fn(self):
        return detr_collate_fn
                
    def get_dataloader(self,**kwargs):
        collate_fn = self.get_collate_fn()
        return DataLoader(self,collate_fn=collate_fn,**kwargs)


@hydra.main(config_path="../configs",config_name="test/clevr_question_answering_train_task_dataset")
def test_dataset(cfg):
    print(cfg.pretty())
    dataset = ClevrQuestionAnsweringTrainTask(cfg.task.clevr_question_answering_train,'train')
    dataloader = dataset.get_dataloader(batch_size=2)
    for data in dataloader:
        import ipdb; ipdb.set_trace()

if __name__=='__main__':
    test_dataset()
