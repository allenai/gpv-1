import hydra
from data.coco.synonyms import SYNONYMS
from data.coco.split_coco_captions import AssignCocoCategories, split_data

import utils.io as io


@hydra.main(config_path='../../configs',config_name='data/split_vqa.yaml')
def main(cfg):
    print(cfg.pretty())
    split_data(cfg)

        

if __name__=='__main__':
    main()