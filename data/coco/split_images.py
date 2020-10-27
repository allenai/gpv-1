import os
import glob
import hydra
import random
from tqdm import tqdm

import utils.io as io


@hydra.main(config_path='../../configs',config_name='data/split_coco_images.yaml')
def main(cfg):
    print(cfg.pretty())
    image_dir = os.path.join(cfg.image_dir,cfg.images[cfg.subset])
    # images is list of full paths to images
    images = glob.glob(f'{image_dir}/*.jpg')
    image_ids = []
    for image in images:
        image_id = int(os.path.split(image)[-1].rstrip('.jpg').split('_')[-1])
        image_ids.append(image_id)

    if cfg.subset=='train':
        N = len(image_ids)
        N_train = int(N*cfg.train_frac)
        random.seed(cfg.seed)
        random.shuffle(image_ids)
        train_image_ids = image_ids[:N_train]
        train_split = {
            'subset': cfg.images[cfg.subset],
            'image_ids': train_image_ids
        }
        io.dump_json_object(
            train_split,
            os.path.join(cfg.exp_dir,'train_images.json'))
        print('Train images:',len(train_split['image_ids']))
        
        val_image_ids = image_ids[N_train:]
        val_split = {
            'subset': cfg.images[cfg.subset],
            'image_ids': val_image_ids
        }
        io.dump_json_object(
            val_split,
            os.path.join(cfg.exp_dir,'val_images.json'))
        print('Val images:',len(val_split['image_ids']))

    elif cfg.subset=='val':
        test_split = {
            'subset': cfg.images[cfg.subset],
            'image_ids': image_ids
        }
        io.dump_json_object(
            test_split,
            os.path.join(cfg.exp_dir,'test_images.json'))
        print('Test images:',len(test_split['image_ids']))
    else:
        raise NotImplementedError


if __name__=='__main__':
    main()