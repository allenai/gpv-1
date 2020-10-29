import os
import hydra
from tqdm import tqdm
from collections import Counter

import utils.io as io


@hydra.main(config_path='../../configs',config_name='data/preprocess_vqa.yaml')
def main(cfg):
    print(cfg.pretty())
    
    subset = cfg.subset
    
    # questions is a dict with keys:
    # 'info', 'task_type', 'data_type', 'license', 'data_subtype', 'questions'
    # 'task_type': 'Open-Ended'
    # 'data_type': 'mscoco'
    # 'data_subtype': 'train2014'
    # 'questions': list of dicts. Eg. {'image_id': 458752, 'question': 'What is this photo taken looking through?', 'question_id': 458752000}
    questions = io.load_json_object(os.path.join(
        f'{cfg.download_dir}/questions',
        cfg.questions[subset]))

    # annotations is a dict with keys:
    # 'info', 'license', 'data_subtype', 'annotations', 'data_type'
    # 'annotations': list of dicts. Eg. {'question_type': 'what is this', 'multiple_choice_answer': 'net', 'answers': [{'answer': 'net', 'answer_confidence': 'maybe', 'answer_id': 1}, {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 2}, {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 3}, {'answer': 'netting', 'answer_confidence': 'yes', 'answer_id': 4}, {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 5}, {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 6}, {'answer': 'mesh', 'answer_confidence': 'maybe', 'answer_id': 7}, {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 8}, {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 9}, {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 10}], 'image_id': 458752, 'answer_type': 'other', 'question_id': 458752000}
    annos = None
    if subset in cfg.anno:
        annos = io.load_json_object(os.path.join(
            f'{cfg.download_dir}/anno',
            cfg.anno[subset]))

    dataset = []
    for i, question_info in enumerate(tqdm(questions['questions'])):
        sample = {
            'query': question_info['question'],
            'question_id': question_info['question_id'],
            'image': {
                'subset': questions['data_subtype'],
                'image_id':  question_info['image_id']
            }
        }
        if annos is not None:
            anno = annos['annotations'][i]
            err_msg = 'question-anno mismatch'
            assert(sample['question_id']==anno['question_id'] and \
                sample['image']['image_id']==anno['image_id']), err_msg
            sample['answer'] = anno['multiple_choice_answer']
            sample['all_answers'] = Counter([a['answer'] for a in anno['answers']])
            sample['anno'] = {
                'question_type': anno['question_type'],
                'answer_type': anno['answer_type']
            }

        dataset.append(sample)

    io.dump_json_object(dataset,os.path.join(cfg.exp_dir,f'{subset}.json'))

if __name__=='__main__':
    main()