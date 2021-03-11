import os
import h5py
import hydra
import json
from collections import Counter

import utils.io as io

task_to_id = {
    'CocoVqa': 'question_id',
    'CocoClassification': 'id',
    'CocoCaptioning': 'cap_id',
    'CocoDetection': 'id',
    'RefCocop': 'sent_id',
}



@hydra.main(config_path=f'../../configs',config_name=f"exp/vis")
def main(cfg):
    boxes_path = os.path.join(
        cfg.eval_dir,
        f'{cfg.task_to_vis}_{cfg.subset}_boxes.h5py')
    boxes_f = h5py.File(boxes_path,'r')
    preds = io.load_json_object(os.path.join(
        cfg.eval_dir,
        f'{cfg.task_to_vis}_{cfg.subset}_predictions.json'))
    preds = preds[:2]
    print(preds)
    boxes_f.close()



if __name__=='__main__':
    main()