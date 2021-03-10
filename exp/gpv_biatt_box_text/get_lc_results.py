import os
import hydra
import json
from collections import Counter

import utils.io as io

@hydra.main(config_path=f'../../configs',config_name=f"exp/lc_results")
def main(cfg):
    print(cfg.tasks)
    print(cfg.results_dir)
    perf = {}
    for task in cfg.tasks:
        json_path = os.path.join(
            cfg.results_dir,
            f'{task}_{cfg.subset}_metrics.json')
        metrics = io.load_json_object(json_path)
        metric = 0
        if task=='RefCocop':
            metric = round(100*metrics['everything']['mAP'],2)
        elif task=='CocoVqa':
            metric = metrics['everything']['accuracy']['all']
        elif task=='CocoCaptioning':
            metric = round(metrics['everything']['scores']['Cider'],3)
        elif task=='CocoDetection':
            metric = round(100*metrics['everything']['mAP'],2)
        elif task=='CocoClassification':
            metric = round(100*metrics['everything']['accuracy']['all'],2)
        

        perf[task]=metric
    print(json.dumps(perf,indent=4))

if __name__=='__main__':
    main()