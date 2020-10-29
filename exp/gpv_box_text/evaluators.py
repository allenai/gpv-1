from collections import Counter
from data.coco.synonyms import SYNONYMS

task_to_id = {
    'CocoVqa': 'question_id',
    'CocoClassification': 'id',
    'CocoCaptioning': 'cap_id',
    'CocoDetection': 'id'
}


class CocoEval():
    def __init__(self,samples,predictions,boxes,task):
        self.task = task
        self.task_id_name = task_to_id[self.task]
        self.samples = {str(s[self.task_id_name]):s for s in samples}
        self.predictions = predictions
        self.boxes = boxes
    
    def sample_novelty(self,sample):
        if len(sample['coco_categories']['unseen']) > 0:
            return 'held_out_concepts'

        return 'seen_concepts'


class CocoVqa(CocoEval):
    def __init__(self,samples,predictions,boxes,task='CocoVqa'):
        super().__init__(samples,predictions,boxes,task)

    def evaluate(self,novelty='everything'):
        absent = 0
        correct = {'all':0}
        total = {'all':0}
        for anno_type in ['answer_type','question_type']:
            correct[anno_type] = Counter()
            total[anno_type] = Counter()

        for k,sample in self.samples.items():
            if novelty is not 'everything' and \
                self.sample_novelty(sample)!=novelty:
                continue
                
            if k not in self.predictions:
                absent += 1
                continue

            pred_answer = self.predictions[k]['answer'].lower()
            gt_answers = {k.lower():v for k,v in sample['all_answers'].items()}
            answer_type = sample['anno']['answer_type']
            question_type = sample['anno']['question_type']
            if pred_answer in gt_answers and gt_answers[pred_answer] >= 3:
                correct['all'] += 1
                correct['answer_type'][answer_type] += 1
                correct['question_type'][question_type] += 1
                
            total['all'] += 1
            total['answer_type'][answer_type] += 1
            total['question_type'][question_type] += 1
        
        eps = 1e-6
        accuracy = {'all':round(100*correct['all']/(eps+total['all']),2)}
        accuracy['answer_type'] = {
            a:round(100*correct['answer_type'][a]/(eps+total['answer_type'][a]),2) 
            for a in total['answer_type']}
        accuracy['question_type'] = {
            a:round(100*correct['question_type'][a]/(eps+total['question_type'][a]),2) 
            for a in total['question_type']}
        metric = {
            'correct': correct,
            'total': total,
            'absent': absent,
            'accuracy': accuracy,
        }
    
        return metric


class CocoClassification(CocoEval):
    def __init__(self,samples,predictions,boxes,task='CocoClassification'):
        super().__init__(samples,predictions,boxes,task)

    def evaluate(self,novelty='everything'):
        absent = 0
        correct = Counter()
        total = Counter()

        for k,sample in self.samples.items():
            if novelty is not 'everything' and \
                self.sample_novelty(sample)!=novelty:
                continue
                
            if k not in self.predictions:
                absent += 1
                continue

            pred_answer = self.predictions[k]['answer'].lower()
            gt_answer =  SYNONYMS[sample['answer']] #[sample['answer']]
            if pred_answer in gt_answer:
                correct['all'] += 1
                correct[sample['answer']] += 1

            total['all'] += 1
            total[sample['answer']] += 1

        eps = 1e-6
        accuracy = {k:round(100*correct[k]/(eps+total[k]),2) for k in total}
        metric = {
            'correct': correct,
            'total': total,
            'absent': absent,
            'accuracy': accuracy,
        }

        return metric