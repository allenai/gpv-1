import os
import h5py
import hydra
import numpy as np
import json
import random
import cv2 as cv
from collections import Counter

import utils.io as io
from utils.html_writer import HtmlWriter

task_to_id_name = {
    'CocoVqa': 'question_id',
    'CocoClassification': 'id',
    'CocoCaptioning': 'cap_id',
    'CocoDetection': 'id',
    'RefCocop': 'sent_id',
}

task_to_config_name = {
    'CocoVqa': 'coco_vqa',
    'CocoClassification': 'coco_classification',
    'CocoCaptioning': 'coco_captioning',
    'CocoDetection': 'coco_detection',
    'RefCocop': 'refcocop',
}


def add_box(img,box_ncxcywh,relevance,color=(200,0,200),thickness=2):
    H,W = img.shape[:2]
    cx,cy,w,h = box_ncxcywh
    x1 = W*(cx-0.5*w)
    y1 = H*(cy-0.5*h)
    x2 = W*(cx+0.5*w)
    y2 = H*(cy+0.5*h)
    x1,y1,x2,y2 = [int(v) for v in [x1,y1,x2,y2]]
    
    cv.rectangle(img,(x1,y1),(x2,y2),color,thickness)
    if relevance is not None:
        cv.rectangle(img,(x1,y1-13),(x1+40,y1-2),(10,10,10),-1)
        cv.putText(img,str(round(relevance,2)),(x1+2,y1-2),cv.FONT_HERSHEY_PLAIN,1,
            (255,0,255),1,cv.LINE_AA)
    else:
        cv.rectangle(img,(x1,y2+2),(x1+30,y2+13),(10,10,10),-1)
        cv.putText(img,'GT',(x1+2,y2+13),cv.FONT_HERSHEY_PLAIN,1,
            (0,255,0),1,cv.LINE_AA)

def vis_sample(sample,out_path):
    img = cv.imread(sample['image_path'])
    if 'crop_box' in sample:
        x,y,w,h = [int(v) for v in sample['crop_box']]
        img = img[y:y+h,x:x+w]

    H,W = img.shape[:2]
    if 'boxes' in sample:
        boxes = sample['boxes']
        B = len(boxes)
        for b in range(B):
            try:
                x,y,w,h = boxes[b]
            except:
                import ipdb; ipdb.set_trace()
            box = [(x + 0.5*w)/W,(y+0.5*h)/H,w/W,h/H]
            add_box(img,box,None,(0,220,0),2)

    boxes = sample['pred']['boxes']
    relevance = sample['pred']['relevance']
    B = boxes.shape[0]
    for b in range(B-1,-1,-1):
        add_box(img,boxes[b],relevance[b])
    
    cv.imwrite(out_path,img)


@hydra.main(config_path=f'../../configs',config_name=f"exp/vis")
def main(cfg):
    index = io.load_json_object(cfg.index)
    if 'RefCocop' in cfg.task_to_vis:
        index = {k:v for k,v in index.items() if 'refcocop' in v}
    random.seed(cfg.seed)
    images_to_vis = random.sample(list(index.keys()),10)

    html_meta = {}
    for task in cfg.task_to_vis:
        boxes_path = os.path.join(
            cfg.eval_dir,
            f'{task}_{cfg.subset}_boxes.h5py')
        boxes_f = h5py.File(boxes_path,'r')
        preds = io.load_json_object(os.path.join(
            cfg.eval_dir,
            f'{task}_{cfg.subset}_predictions.json'))
        task_name = task_to_config_name[task]
        id_name = task_to_id_name[task]
        samples = io.load_json_object(
            cfg.task_configs[task_name]['samples'][cfg.subset])
        
        for image_to_vis in images_to_vis:
            if task_name not in index[image_to_vis]:
                continue

            sample_ids = index[image_to_vis][task_name]
            html_dir = os.path.join(cfg.vis_dir,image_to_vis)
            if html_dir not in html_meta:
                html_meta[html_dir] = {}
            
            io.mkdir_if_not_exists(html_dir,recursive=True)
            for sample_id in sample_ids:
                sample = samples[sample_id]
                if task=='CocoClassification':
                    sample['crop_box'] = sample['boxes']
                    sample.pop('boxes')

                boxes = boxes_f[str(sample[id_name])]['boxes'][()]
                relevance = boxes_f[str(sample[id_name])]['relevance'][()]
                if 'boxes' in sample:
                    B = max(np.sum(relevance > 0.8),len(sample['boxes']))
                else:
                    B = max(np.sum(relevance > 0.8),5)
                
                if task == 'RefCocop': 
                    B=len(sample['boxes'])

                sample['pred'] = {
                    'text':preds[str(sample[id_name])],
                    'boxes': boxes[:B],
                    'relevance': relevance[:B]
                }
                image_id = sample['image']['image_id']
                image_subset = sample['image']['subset']
                sample['image_path'] = os.path.join(
                    cfg.task_configs[task_name].image_dir,
                    f'{image_subset}/COCO_{image_subset}_' + str(image_id).zfill(12) + '.jpg')
                
                original_img_path = os.path.join(html_dir,'original.png')
                if not os.path.exists(original_img_path):
                    img = cv.imread(sample['image_path'])
                    cv.imwrite(original_img_path,img)

                html_meta[html_dir]['original.png'] = None
                out_img_name = f'{task}_{sample_id}.png'
                out_path = os.path.join(html_dir,out_img_name)
                vis_sample(sample,out_path)
                html_meta[html_dir][out_img_name] = {
                    'query': sample['query'],
                    'text': sample['pred']['text'],
                    'relevance': [round(v,2) for v in relevance[:B]]
                }

        boxes_f.close()
    
    for html_dir, info in html_meta.items():
        html_writer = HtmlWriter(os.path.join(html_dir,'index.html'))
        for img_name in info.keys():
            if info[img_name] is None:
                col_dict = {
                    0: '',
                    1: '',
                    2: html_writer.image_tag_original_size(img_name),
                    3: ''
                }
            else:
                col_dict = {
                    0: info[img_name]['query'],
                    1: info[img_name]['text'],
                    2: html_writer.image_tag_original_size(img_name),
                    3: info[img_name]['relevance']
                }
            html_writer.add_element(col_dict)
        
        html_writer.close()



if __name__=='__main__':
    main()