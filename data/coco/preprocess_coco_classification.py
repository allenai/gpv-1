import os
import hydra
import random
from tqdm import tqdm

import utils.io as io


@hydra.main(config_path='../../configs',config_name='data/preprocess_coco_classification.yaml')
def main(cfg):
    print(cfg.pretty())
    subset = cfg.subset

    query_choices = [
        'What is this?',
        'What is this object?',
        'What object is this?',
        'What is this thing?'
    ]

    # data is a dict with keys:
    # 'info', 'images', 'licenses', 'annotations', 'categories'
    data = io.load_json_object(
        os.path.join(cfg.download_dir,cfg.instances[subset]))
    
    instances = {}
    instance_ids = {}
    for anno in tqdm(data['annotations']):
        image_id = anno['image_id']
        category_id = anno['category_id']
        if image_id not in instances:
            instances[image_id] = {}
            instance_ids[image_id] = {}

        if category_id not in instances[image_id]:
            instances[image_id][category_id] = []

        instances[image_id][category_id].append(anno['bbox'])
        if category_id in instance_ids[image_id]:
            assert instance_ids[image_id][category_id] == anno['id']
        instance_ids[image_id][category_id] = anno['id']

    categories = {}
    for category in tqdm(data['categories']):
        category_id = category['id']
        categories[category_id] = category

    images = {
        image_info['id']: image_info for image_info in data['images']}

    dataset = []
    for image_id, category_boxes in tqdm(instances.items()):
        for category_id, boxes in category_boxes.items():
            category = categories[category_id]
            image_path = images[image_id]['file_name'] 
            sample = {
                'query': random.choice(query_choices),
                'boxes': random.choice(boxes),
                'category_id': category_id,
                'answer': category['name'],
                'image': {
                    'subset': image_path.split('_')[1],
                    'image_id': image_id
                }
                'id': instance_ids[image_id][category_id]
            }
            dataset.append(sample)

    io.dump_json_object(dataset,os.path.join(cfg.exp_dir,f'{subset}.json'))

if __name__=='__main__':
    main()
