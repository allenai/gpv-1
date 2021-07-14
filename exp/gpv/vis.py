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