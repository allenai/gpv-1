import os
import hydra
from tqdm import tqdm

import utils.io as io


def generate_single_attribute_query_detections(scene_info,attribute_type='shape'):
    attributes = set()
    for obj_info in scene_info['objects']:
        attributes.add(obj_info[attribute_type])

    query_to_object_ids = {}
    for attribute in attributes:
        obj_ids = []
        for i,obj_info in enumerate(scene_info['objects']):
            if obj_info[attribute_type]==attribute:
                obj_ids.append(i)
        
        query_to_object_ids[f'{attribute_type} is {attribute}'] = obj_ids
    
    return query_to_object_ids


def create_samples(query_to_object_ids, scene_info):
    samples = []
    objects = scene_info['objects']
    for query, obj_ids in query_to_object_ids.items():
        samples.append({
            'image': scene_info['image_filename'],
            'query': query,
            'objects': [objects[i] for i in obj_ids]
        })

    return samples


@hydra.main(config_path=f'../../configs',config_name=f"data/generate_query_detection_data.yaml")
def main(cfg):
    data_cfg = cfg.task.clevr_detection_train[cfg.subset]
    dataset = []
    for i in tqdm(range(data_cfg.num_images)):
        scene_json = os.path.join(
            data_cfg.scenes,
            'CLEVR_new_'+str(i).zfill(6)+'.json')
        scene_info = io.load_json_object(scene_json) 

        for attribute_type in ['shape','color']:
            query_to_object_ids = generate_single_attribute_query_detections(
                scene_info,
                attribute_type=attribute_type)

            samples = create_samples(query_to_object_ids,scene_info)

            dataset.extend(samples)
        
    print('Num samples:',len(dataset))
    io.dump_json_object(dataset,os.path.join(cfg.output_dir,'dataset.json'))
    



if __name__=='__main__':
    main()