import os
import h5py
import hydra
import numpy as np
import json
import cv2 as cv
from tqdm import tqdm
from collections import Counter

import utils.io as io


@hydra.main(config_path=f'../configs',config_name=f"data/create_index")
def main(cfg):
    index = {}
    for task in cfg.tasks:
        samples = io.load_json_object(
            cfg.task_configs[task]['samples'][cfg.subset])
        print(task)
        for i,sample in enumerate(tqdm(samples)):
            imid = sample['image']['image_id']
            imsubset = sample['image']['subset']
            imname = 'COCO_' + imsubset + '_' + str(imid).zfill(12)
            if imname not in index:
                index[imname] = {}
            
            if task not in index[imname]:
                index[imname][task] = []

            index[imname][task].append(i)

    outpath = os.path.join(cfg.exp_dir,'index.json')
    io.dump_json_object(index,outpath)
        

if __name__=='__main__':
    main()