import os
import hydra
import random
from tqdm import tqdm

import utils.io as io


@hydra.main(config_path='../../configs',config_name='data/preprocess_coco_captions.yaml')
def main(cfg):
    print(cfg.pretty())
    subset = cfg.subset

    query_choices = [
        'Generate a caption.',
        'Generate a description.',
        'Describe this image.',
        'Describe the image.',
        'Caption this image.',
        'Caption the image.',
        'What is happening in this image.',
        'What is happening in the image.',
        'What is going on in this image.',
        'What is going on in the image.',
        'Generate a caption for this image.',
        'Generate a caption for the image.',
        'Generate a description for this image.',
        'Generate a description for the image.',
    ]
    
    if subset=='test':
        images = io.load_json_object(os.path.join(
            cfg.download_dir,cfg.captions[subset]))['images']
        dataset = []
        for i, image_info in enumerate(tqdm(images)):
            image_id = image_info['id']
            image_path = image_info['file_name'] 
            sample = {
                'query': random.choice(query_choices),
                'image': {
                    'subset': image_path.split('_')[1],
                    'image_id': image_id
                }
            }
            dataset.append(sample)
    
    else:
        # data consists of the following keys:
        # 'info', 'licenses', 'images', 'annotations'
        # 'images': list of dict. Eg. {'license': 5, 'file_name': 'COCO_train2014_000000057870.jpg', 'coco_url': 'http://images.cocodataset.org/train2014/COCO_train2014_000000057870.jpg', 'height': 480, 'width': 640, 'date_captured': '2013-11-14 16:28:13', 'flickr_url': 'http://farm4.staticflickr.com/3153/2970773875_164f0c0b83_z.jpg', 'id': 57870}
        # 'annotations': list of dict. Eg. {'image_id': 318556, 'id': 48, 'caption': 'A very clean and well decorated empty bathroom'}
        data = io.load_json_object(os.path.join(
            cfg.download_dir,cfg.captions[subset]))
        images = {
            image_info['id']: image_info for image_info in data['images']}
        dataset = []
        for i, caption_info in enumerate(tqdm(data['annotations'])):
            image_id = caption_info['image_id']
            cap_id = caption_info['id']
            image_path = images[image_id]['file_name'] 
            sample = {
                'query': random.choice(query_choices),
                'answer': caption_info['caption'],
                'cap_id': cap_id,
                'image': {
                    'subset': image_path.split('_')[1],
                    'image_id': image_id
                }
            }
            dataset.append(sample)

    io.dump_json_object(dataset,os.path.join(cfg.exp_dir,f'{subset}.json'))

if __name__=='__main__':
    main()