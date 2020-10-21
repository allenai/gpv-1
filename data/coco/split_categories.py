import os
import hydra
import random
from tqdm import tqdm

import utils.io as io


@hydra.main(config_path='../../configs',config_name='data/split_coco_categories.yaml')
def main(cfg):
    print(cfg.pretty())
    categories = io.load_json_object(
        os.path.join(cfg.download_dir,cfg.instances[cfg.subset]))['categories']
    
    person_category = [c for c in categories if c['name']=='person']
    nonperson_category = [c for c in categories if c['name']!='person']
    num_categories = len(categories)
    
    random.seed(cfg.seed)
    random.shuffle(nonperson_category)
    shared_categories = person_category
    shared_categories.extend(nonperson_category[:cfg.num_categories.shared-1])
    categories_held_from_vqa = nonperson_category[
        cfg.num_categories.shared-1:cfg.num_categories.shared-1+cfg.num_categories.held_from_vqa]
    categories_held_from_det = nonperson_category[
        cfg.num_categories.shared-1+cfg.num_categories.held_from_vqa:]

    category_split = {
        'shared': shared_categories,
        'held_from_vqa': categories_held_from_vqa,
        'held_from_det': categories_held_from_det,
        'held_from_cap': categories_held_from_vqa,
        'held_from_cls': categories_held_from_det,
    }
    io.dump_json_object(
        category_split,
        os.path.join(cfg.exp_dir,'category_split.json'))


if __name__=='__main__':
    main()