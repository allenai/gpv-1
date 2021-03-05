import h5py
import os
from tqdm import tqdm
import imagesize

from data.coco.synonyms import SYNONYMS
from .frcnn_classes import FRCNN_CLASSES
import utils.io as io


def category_name_to_frcnn_id_dict():
    mapping = {}
    for cat_name in SYNONYMS:
        mapping[cat_name] = FRCNN_CLASSES.index(cat_name)

    return mapping


def main():
    outdir = "/home/tanmayg/Data/gpv/coco_exp/loc_baseline_original_split/eval"
    io.mkdir_if_not_exists(outdir,recursive=True)

    subset='test'
    samples_json = f"/home/tanmayg/Data/gpv/learning_phase_data/coco_detection/gpv_split/{subset}.json"
    if subset=='val':
        prefix = 'COCO_train2014_'
        feats_path = '/home/tanmayg/Data/bua_very_greedy/train2014.hdf5'
        imgdir = '/home/tanmayg/Data/gpv/learning_phase_data/coco/images/train2014'
    elif subset=='test':
        prefix = 'COCO_val2014_'
        feats_path = '/home/tanmayg/Data/bua_very_greedy/val2014.hdf5'
        imgdir = '/home/tanmayg/Data/gpv/learning_phase_data/coco/images/val2014'

    cat_name_to_frcnn_cls_id = category_name_to_frcnn_id_dict()

    samples = io.load_json_object(samples_json)
    feats_file = h5py.File(feats_path,'r')
    predictions = {}
    boxes_path = os.path.join(outdir,f'CocoDetection_{subset}_boxes.h5py')
    boxes_h5py = h5py.File(boxes_path,'w')
    for i,sample in enumerate(tqdm(samples)):
        category_name = sample['category_name']
        category_id = sample['category_id']
        det_id = sample['id']
        frcnn_cls_id = cat_name_to_frcnn_cls_id[category_name]
        imgid = sample['image']['image_id']
        imgname = prefix + str(imgid).zfill(12)
        imgpath = os.path.join(imgdir,imgname+'.jpg')
        W,H = imagesize.get(imgpath)
        scores = feats_file[imgname]['scores'][()]
        boxes = feats_file[imgname]['boxes'][()] # x1,y1,x2,y2
        classes = feats_file[imgname]['classes'][()]
        
        sample_id = str(det_id)
        predictions[sample_id] = {'answer': category_name}
        sel_ids = classes==frcnn_cls_id
        sel_scores = scores[sel_ids]
        sel_boxes = boxes[sel_ids]
        
        #normalize to 0,1
        sel_boxes[:,0] = sel_boxes[:,0]/W
        sel_boxes[:,2] = sel_boxes[:,2]/W
        sel_boxes[:,1] = sel_boxes[:,1]/H
        sel_boxes[:,3] = sel_boxes[:,3]/H

        # convert to cxcywh
        box_w = sel_boxes[:,2]-sel_boxes[:,0]
        box_h = sel_boxes[:,3]-sel_boxes[:,1]
        sel_boxes[:,0] = sel_boxes[:,0] + 0.5*box_w
        sel_boxes[:,1] = sel_boxes[:,1] + 0.5*box_h
        sel_boxes[:,2] = box_w
        sel_boxes[:,3] = box_h

        grp = boxes_h5py.create_group(sample_id)
        grp.create_dataset('boxes',data=sel_boxes)
        grp.create_dataset('relevance',data=sel_scores)

    boxes_h5py.close()
    pred_path = os.path.join(outdir,f'CocoDetection_{subset}_predictions.json')
    io.dump_json_object(predictions,pred_path)

if __name__=='__main__':
    main()
        

