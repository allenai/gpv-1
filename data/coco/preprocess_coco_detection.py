import os
import hydra
import random
from tqdm import tqdm

import utils.io as io


@hydra.main(config_path='../../configs',config_name='data/preprocess_coco_detection.yaml')
def main(cfg):
    print(cfg.pretty())
    subset = cfg.subset

    query_choices = [
        'Locate {}.',
        'Locate {} in the image.',
        'Locate {} in this image.',
        'Locate instances of {}.',
        'Locate instances of {} in the image.',
        'Locate instances of {} in this image.',
        'Locate all instances of {}.',
        'Locate all instances of {} in the image.',
        'Locate all instances of {} in this image.',
        'Find {}.',
        'Find {} in the image.',
        'Find {} in this image.',
        'Find instances of {}.',
        'Find instances of {} in the image.',
        'Find instances of {} in this image.',
        'Find all instances of {}.',
        'Find all instances of {} in the image.',
        'Find all instances of {} in this image.',
    ]

    # data is a dict with keys:
    # 'info', 'images', 'licenses', 'annotations', 'categories'
    data = io.load_json_object(
        os.path.join(cfg.download_dir,cfg.instances[subset]))
    
    instances = {}
    for anno in tqdm(data['annotations']):
        image_id = anno['image_id']
        category_id = anno['category_id']
        if image_id not in instances:
            instances[image_id] = {}

        if category_id not in instances[image_id]:
            instances[image_id][category_id] = []

        instances[image_id][category_id].append(anno['bbox'])
    
    categories = {}
    for category in tqdm(data['categories']):
        category_id = category['id']
        category_name = category['name']
        categories[category_id] = category_name

    images = {
        image_info['id']: image_info for image_info in data['images']}

    dataset = []
    for image_id, category_boxes in tqdm(instances.items()):
        for category_id, boxes in category_boxes.items():
            category_name = categories[category_id]
            image_path = images[image_id]['file_name'] 
            sample = {
                'query': random.choice(query_choices).format(category_name),
                'boxes': boxes,
                'category_id': category_id,
                'category_name': category_name,
                'image': {
                    'subset': image_path.split('_')[1],
                    'image_id': image_id
                }
            }
            dataset.append(sample)

    io.dump_json_object(dataset,os.path.join(cfg.exp_dir,f'{subset}.json'))

if __name__=='__main__':
    main()