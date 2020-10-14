import os
import hydra
from tqdm import tqdm
from collections import Counter

import utils.io as io


def generate_questions(scene_info,query_to_object_id,attribute_type='material'):
    samples = []
    objects = scene_info['objects']
    for query, obj_id in query_to_object_id.items():
        obj_info = objects[obj_id]
        samples.append({
            'image': scene_info['image_filename'],
            'query': f'what is the {attribute_type} of {query}',
            'objects': [obj_info],
            'answer': obj_info[attribute_type],
        })
    
    return samples


def find_unique_object_queries(scene_info,query_attributes):
    queries = [None]*len(scene_info['objects'])
    cnt = Counter()
    for i, obj_info in enumerate(scene_info['objects']):
        queries[i] = ' '.join(
            [obj_info[attribute] for attribute in query_attributes])
        cnt[queries[i]] += 1
    
    query_to_object_id = {}
    for i,query in enumerate(queries):
        if cnt[query]==1:
            query_to_object_id[query] = i
    
    return query_to_object_id


@hydra.main(
    config_path=f'../../configs',
    config_name=f"data/generate_question_answering_data.yaml")
def main(cfg):
    data_cfg = cfg.task.clevr_detection_train[cfg.subset]
    dataset = []
    for i in tqdm(range(data_cfg.num_images)):
        scene_json = os.path.join(
            data_cfg.scenes,
            'CLEVR_new_'+str(i).zfill(6)+'.json')
        scene_info = io.load_json_object(scene_json) 

        query_to_object_id = find_unique_object_queries(
            scene_info,
            cfg.query_attributes)
        
        for attribute_type in cfg.answer_attributes:
            samples = generate_questions(
                scene_info,
                query_to_object_id,
                attribute_type)
            
            dataset.extend(samples)
        
    print('Num samples:',len(dataset))
    io.dump_json_object(dataset,os.path.join(cfg.output_dir,'dataset.json'))
    



if __name__=='__main__':
    main()