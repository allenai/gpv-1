import numpy as np
from collections import Counter
from tqdm import tqdm
from data.coco.synonyms import SYNONYMS
from third_party.pycocoevalcap.eval import *
import third_party.detection_metrics.lib.Evaluator as det_evaluator

task_to_id = {
    'CocoVqa': 'question_id',
    'CocoClassification': 'id',
    'CocoCaptioning': 'cap_id',
    'CocoDetection': 'id',
    'RefCocop': 'sent_id',
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
            if pred_answer in gt_answers:
                correctness = min(gt_answers[pred_answer]/3,1)
                correct['all'] += correctness
                correct['answer_type'][answer_type] += correctness
                correct['question_type'][question_type] += correctness
                
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
        metrics = {
            'correct': correct,
            'total': total,
            'absent': absent,
            'accuracy': accuracy,
        }
    
        return metrics


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
        accuracy = {k:round(correct[k]/(eps+total[k]),4) for k in total}
        metrics = {
            'correct': correct,
            'total': total,
            'absent': absent,
            'accuracy': accuracy,
        }

        return metrics


class CocoCaptioning(CocoEval):
    def __init__(self,samples,predictions,boxes,task='CocoCaptioning'):
        super().__init__(samples,predictions,boxes,task)
        self.subset_imgid2gtcaps = {}
        self.tokenizer = PTBTokenizer()
        for sample in samples:
            imgid = str(sample['image']['image_id']).zfill(12)
            subset = sample['image']['subset']
            subset_imgid = f'{subset}_{imgid}'
            if subset_imgid not in self.subset_imgid2gtcaps:
                self.subset_imgid2gtcaps[subset_imgid] = []
            self.subset_imgid2gtcaps[subset_imgid].append(sample['answer'].lower())
                #[w.lower() for w in self.tokenizer.tokenize(sample['answer'])])
        self.scorers = {
            'Bleu': Bleu(4),
            # 'Meteor': Meteor(),
            # 'Rouge': Rouge(),
            'Cider': Cider(),
            # 'Spice': Spice()
        }

        
    def evaluate(self,novelty='everything'):
        absent = 0

        refs = {}
        hyps = {}
        for k,sample in tqdm(self.samples.items()):
            if novelty is not 'everything' and \
                self.sample_novelty(sample)!=novelty:
                continue
                
            if k not in self.predictions:
                absent += 1
                continue
            
            imgid = str(sample['image']['image_id']).zfill(12)
            subset = sample['image']['subset']
            subset_imgid = f'{subset}_{imgid}'

            #pred_answer = [w.lower() for w in 
            #    self.tokenizer.tokenize(self.predictions[k]['answer'])]
            pred_answer = self.predictions[k]['answer'].lower()
            gt_answers = self.subset_imgid2gtcaps[subset_imgid]
            cap_id = sample['cap_id']
            refs[cap_id] = []
            for c in gt_answers:
                refs[cap_id].append({'caption':c})

            hyps[cap_id] = [{'caption':pred_answer}]
        
        is_empty = (len(hyps)==0)
        if not is_empty:
            refs = self.tokenizer.tokenize(refs)
            hyps = self.tokenizer.tokenize(hyps)
            
        metrics = {
            'absent': absent,
            'total': len(hyps),
            'scores': {}
        }
        print(metrics)
        for metric,scorer in self.scorers.items():
            if is_empty is True:
                if metric=='Bleu':
                    scores = [0]*4
                else:
                    scores = 0
            else:
                scores = scorer.compute_score(refs,hyps)[0]

            if metric=='Bleu':
                for i,score in enumerate(scores):
                    metrics['scores'][metric+str(i+1)] = score
            else:
                metrics['scores'][metric] = scores

        return metrics


class CocoDetection(CocoEval):
    def __init__(self,samples,predictions,boxes,task='CocoDetection'):
        super().__init__(samples,predictions,boxes,task)

    def evaluate(self,novelty='everything',iou_thresh=0.5):
        absent = 0
        correct = Counter()
        total = Counter()
        APs = []
        eval_engine = det_evaluator.Evaluator()
        for k,sample in tqdm(self.samples.items()):
            if novelty is not 'everything' and \
                self.sample_novelty(sample)!=novelty:
                continue
                
            if k not in self.predictions:
                absent += 1
                continue

            scores = self.boxes[k]['relevance'][()] # probs
            pred_boxes = self.boxes[k]['boxes'][()] # scaled 0 to 1 (cx,cy,w,h)
            # convert to (x,y,w,h)
            pred_boxes[:,0] = pred_boxes[:,0] - 0.5*pred_boxes[:,2]
            pred_boxes[:,1] = pred_boxes[:,1] - 0.5*pred_boxes[:,3]
            gt_boxes = np.array(sample['boxes']) # absolute coord (x,y,w,h)
            # convert to relative coordinates
            W = sample['image']['W']
            H = sample['image']['H']
            gt_boxes[:,0] = gt_boxes[:,0]/W
            gt_boxes[:,1] = gt_boxes[:,1]/H
            gt_boxes[:,2] = gt_boxes[:,2]/W
            gt_boxes[:,3] = gt_boxes[:,3]/H
            B = pred_boxes.shape[0]
            all_boxes = det_evaluator.BoundingBoxes()
            for b in range(B):
                x,y,w,h=pred_boxes[b]
                all_boxes.addBoundingBox(det_evaluator.BoundingBox(
                    imageName=sample['image']['image_id'],
                    classId=sample['category_name'],
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    typeCoordinates=det_evaluator.CoordinatesType.Relative,
                    imgSize=(W,H),
                    bbType=det_evaluator.BBType.Detected,
                    classConfidence=scores[b],
                    format=det_evaluator.BBFormat.XYWH))

            B = gt_boxes.shape[0]
            for b in range(B):
                x,y,w,h = gt_boxes[b]
                all_boxes.addBoundingBox(det_evaluator.BoundingBox(
                    imageName=sample['image']['image_id'],
                    classId=sample['category_name'],
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    typeCoordinates=det_evaluator.CoordinatesType.Relative,
                    imgSize=(W,H),
                    bbType=det_evaluator.BBType.GroundTruth,
                    format=det_evaluator.BBFormat.XYWH))
            
            det_metrics = eval_engine.GetPascalVOCMetrics(all_boxes,iou_thresh)
            APs.append(det_metrics[0]['AP'])
            total['all'] += 1
            total[sample['category_name']] += 1

        mAP = np.mean(APs)
        
        metrics = {
            'absent': absent,
            'total': total,
            'mAP': mAP
        }

        return metrics

class RefCocop(CocoEval):
    def __init__(self,samples,predictions,boxes,task='RefCocop'):
        super().__init__(samples,predictions,boxes,task)

    def evaluate(self,novelty='everything',iou_thresh=0.5):
        absent = 0
        correct = Counter()
        total = Counter()
        APs = []
        eval_engine = det_evaluator.Evaluator()
        for k,sample in tqdm(self.samples.items()):
            if novelty is not 'everything' and \
                self.sample_novelty(sample)!=novelty:
                continue
                
            if k not in self.predictions:
                absent += 1
                continue

            scores = self.boxes[k]['relevance'][()] # probs
            pred_boxes = self.boxes[k]['boxes'][()] # scaled 0 to 1 (cx,cy,w,h)
            # convert to (x,y,w,h)
            pred_boxes[:,0] = pred_boxes[:,0] - 0.5*pred_boxes[:,2]
            pred_boxes[:,1] = pred_boxes[:,1] - 0.5*pred_boxes[:,3]
            gt_boxes = np.array(sample['boxes']) # absolute coord (x,y,w,h)
            # convert to relative coordinates
            W = sample['image']['W']
            H = sample['image']['H']
            gt_boxes[:,0] = gt_boxes[:,0]/W
            gt_boxes[:,1] = gt_boxes[:,1]/H
            gt_boxes[:,2] = gt_boxes[:,2]/W
            gt_boxes[:,3] = gt_boxes[:,3]/H
            B = pred_boxes.shape[0]
            all_boxes = det_evaluator.BoundingBoxes()
            for b in range(B):
                x,y,w,h=pred_boxes[b]
                all_boxes.addBoundingBox(det_evaluator.BoundingBox(
                    imageName=sample['image']['image_id'],
                    classId='query',
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    typeCoordinates=det_evaluator.CoordinatesType.Relative,
                    imgSize=(W,H),
                    bbType=det_evaluator.BBType.Detected,
                    classConfidence=scores[b],
                    format=det_evaluator.BBFormat.XYWH))

            B = gt_boxes.shape[0]
            for b in range(B):
                x,y,w,h = gt_boxes[b]
                all_boxes.addBoundingBox(det_evaluator.BoundingBox(
                    imageName=sample['image']['image_id'],
                    classId='query',
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    typeCoordinates=det_evaluator.CoordinatesType.Relative,
                    imgSize=(W,H),
                    bbType=det_evaluator.BBType.GroundTruth,
                    format=det_evaluator.BBFormat.XYWH))
            
            det_metrics = eval_engine.GetPascalVOCMetrics(all_boxes,iou_thresh)
            APs.append(det_metrics[0]['AP'])
            total['all'] += 1

        mAP = np.mean(APs)
        
        metrics = {
            'absent': absent,
            'total': total,
            'mAP': mAP
        }

        return metrics

# class CocoDetection(CocoEval):
#     def __init__(self,samples,predictions,boxes,task='CocoDetection'):
#         super().__init__(samples,predictions,boxes,task)

#     def evaluate(self,novelty='everything',iou_thresh=0.5):
#         absent = 0
#         correct = Counter()
#         total = Counter()

#         all_boxes = det_evaluator.BoundingBoxes()
#         for k,sample in tqdm(self.samples.items()):
#             if novelty is not 'everything' and \
#                 self.sample_novelty(sample)!=novelty:
#                 continue
                
#             if k not in self.predictions:
#                 absent += 1
#                 continue

#             scores = self.boxes[k]['relevance'][()] # probs
#             pred_boxes = self.boxes[k]['boxes'][()] # scaled 0 to 1 (cx,cy,w,h)
#             # convert to (x,y,w,h)
#             pred_boxes[:,0] = pred_boxes[:,0] - 0.5*pred_boxes[:,2]
#             pred_boxes[:,1] = pred_boxes[:,1] - 0.5*pred_boxes[:,3]
#             gt_boxes = np.array(sample['boxes']) # absolute coord (x,y,w,h)
#             # convert to relative coordinates
#             W = sample['image']['W']
#             H = sample['image']['H']
#             gt_boxes[:,0] = gt_boxes[:,0]/W
#             gt_boxes[:,1] = gt_boxes[:,1]/H
#             gt_boxes[:,2] = gt_boxes[:,2]/W
#             gt_boxes[:,3] = gt_boxes[:,3]/H
#             B = pred_boxes.shape[0]
#             for b in range(B):
#                 x,y,w,h=pred_boxes[b]
#                 all_boxes.addBoundingBox(det_evaluator.BoundingBox(
#                     imageName=sample['image']['image_id'],
#                     classId=sample['category_name'],
#                     x=x,
#                     y=y,
#                     w=w,
#                     h=h,
#                     typeCoordinates=det_evaluator.CoordinatesType.Relative,
#                     imgSize=(W,H),
#                     bbType=det_evaluator.BBType.Detected,
#                     classConfidence=scores[b],
#                     format=det_evaluator.BBFormat.XYWH))

#             B = gt_boxes.shape[0]
#             for b in range(B):
#                 x,y,w,h = gt_boxes[b]
#                 all_boxes.addBoundingBox(det_evaluator.BoundingBox(
#                     imageName=sample['image']['image_id'],
#                     classId=sample['category_name'],
#                     x=x,
#                     y=y,
#                     w=w,
#                     h=h,
#                     typeCoordinates=det_evaluator.CoordinatesType.Relative,
#                     imgSize=(W,H),
#                     bbType=det_evaluator.BBType.GroundTruth,
#                     format=det_evaluator.BBFormat.XYWH))
            
#             total['all'] += 1
#             total[sample['category_name']] += 1
        
#         if total==0:
#             APs = {}
#             mAP = None
#         else:
#             eval_engine = det_evaluator.Evaluator()
#             det_metrics = eval_engine.GetPascalVOCMetrics(all_boxes,iou_thresh)
#             APs = {m['class']: m['AP'] for m in det_metrics}
#             mAP = np.mean(list(APs.values()))
        
#         metrics = {
#             'absent': absent,
#             'total': total,
#             'AP': APs,
#             'mAP': mAP
#         }

#         return metrics