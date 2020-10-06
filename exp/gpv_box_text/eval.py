import os
import math
import hydra
import torch
import itertools
import numpy as np
import skimage.io as skio
from tqdm import tqdm
from collections import Counter

from .models.gpv import GPV
from datasets.clevr_question_answering_train_task import ClevrQuestionAnsweringTrainTask
from utils.bbox_utils import vis_bbox
import utils.io as io

import third_party.detection_metrics.lib.Evaluator as evaluator

# Doesn't Work Currently


class Metrics():
    def __init__(self):
        self.skill_concept_cnt = Counter()
        self.skill_concept_is_correct = Counter()

    def reset():
        self.skill_concept_cnt = {}
        self.skill_concept_is_correct = {}

    def map_to_skill_concepts(self,queries,targets,answer_vocab):
        batch_skills = []
        batch_concepts = []
        for query, target in zip(queries,targets):
            skills = []
            tokens = query.split(' ')
            if 'boxes' in target:
                if 'answer' in target:
                    skills.append('Rel.'+tokens[0])
                else:
                    skills.append('Loc.'+tokens[0])
            if 'answer' in target:
                skills.append('Ans.'+tokens[0])
            
            batch_skills.append(skills)
            
            concepts = [t for t in tokens if t not in ['of','is']]
            if 'answer' in target:
                concepts = concepts + [answer_vocab[target['answer']]]
            
            batch_concepts.append(concepts)
        
        return batch_skills, batch_concepts
    
    def answer_correctness(self,outputs,targets):
        logits = outputs['answer_logits'][-1]
        pred_answers = torch.topk(logits,k=1,dim=1).indices[:,0].cpu()
        is_correct = [True]*len(targets)
        for i, target in enumerate(targets):
            if 'answer' in target:
                is_correct[i] = pred_answers[i]==target['answer']
        
        return is_correct

    def correctness(self,outputs,targets):
        is_correct = answer_correctness(outputs,targets)
        return is_correct

    def update_metrics(self,outputs,targets):
        is_correct = self.correctness(outputs,targets)
        batch_skills, batch_concepts = self.map_to_skill_concepts(
            queries,targets)
        for i in len(is_correct):
            skills = batch_skills[i]
            concepts = batch_skills[i]
            value = int(is_correct[i])
            for skill, concept in zip(skills,concepts):
                skill_concept = f'{skill}_{concept}'
                self.skill_concept_cnt[skill_concept] += 1
                self.skill_concept_is_correct[skill_concept] += value

    def total_perf(self):
        skill_perf = {}
        concept_perf = {}
        skill_concept_perf = {}
        for skill_concept, cnt in self.skill_concept_cnt.items():
            if cnt==0:
                continue
                
            perf = self.skill_concept_is_correct[skill_concept] / cnt
            skill_concept_perf[skill_concept] = perf
            
            skill,concept = skill_concept.split('_')
            if skill not in skill_perf:
                skill_perf[skill] = []
            
            skill_perf[skill].append(perf)

            if concept not in concept_perf:
                concept_perf[concept] = []
            
            concept_perf[concept].append(perf)
        
        skill_perf = {s:np.mean(ps) for s,ps in skill_perf.items()}
        concept_perf = {c:np.mean(ps) for c,ps in concept_perf.items()}



def eval_model(model,dataloader,cfg,max_samples=None):
    model.eval()
    cnt = 0
    all_boxes = evaluator.BoundingBoxes()
    num_max_iter = None
    if max_samples is not None:
        num_max_iter = math.ceil(max_samples / dataloader.batch_size)
    
    total_correct = 0
    total_samples = 0
    for it,data in enumerate(itertools.islice(tqdm(dataloader),num_max_iter)):
        imgs,queries,targets = data
        imgs = imgs.to(torch.device(0))

        outputs = model(imgs,queries)
        pred_prob = outputs['pred_logits'].softmax(-1).detach().cpu().numpy()
        pred_boxes = outputs['pred_boxes'].detach().cpu().numpy()
        B = pred_boxes.shape[0]
        H = imgs.tensors.size(2)
        W = imgs.tensors.size(3)
        img_size = (W,H)

        correct, total = compute_acc(outputs,targets)
        total_correct += correct
        total_samples += total

    
        for b in range(B):
            img_name = str(cnt).zfill(6)

            gt_boxes = targets[b]['boxes'].numpy()
            for k in range(gt_boxes.shape[0]):
                x,y,w,h = gt_boxes[k]
                x = x - 0.5*w
                y = y - 0.5*h
                all_boxes.addBoundingBox(evaluator.BoundingBox(
                    imageName=img_name,
                    classId='relevant',
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    typeCoordinates=evaluator.CoordinatesType.Relative,
                    imgSize=img_size,
                    bbType=evaluator.BBType.GroundTruth,
                    format=evaluator.BBFormat.XYWH))
            
            boxes = pred_boxes[b]
            probs = pred_prob[b]
            for k in range(boxes.shape[0]):
                x,y,w,h = boxes[k]
                x = x - 0.5*w
                y = y - 0.5*h
                all_boxes.addBoundingBox(evaluator.BoundingBox(
                    imageName=img_name,
                    classId='relevant',
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    typeCoordinates=evaluator.CoordinatesType.Relative,
                    imgSize=img_size,
                    bbType=evaluator.BBType.Detected,
                    classConfidence=probs[k][0],
                    format=evaluator.BBFormat.XYWH))

            cnt+=1

    eval_engine = evaluator.Evaluator()
    metrics = eval_engine.GetPascalVOCMetrics(all_boxes,cfg.eval.iou_thresh)[0]
    metrics['Accuracy'] = total_correct / total_samples
    return metrics


@hydra.main(config_path=f'../../configs',config_name=f"exp/detr_lang_qa")
def main(cfg):
    print(cfg.pretty())

    model = GPV(cfg.model).cuda()

    ckpt_pth = os.path.join(cfg.ckpt_dir,str(cfg.eval.step).zfill(6)+'.pth')
    model.load_state_dict(torch.load(ckpt_pth)['model'])
    
    dataset = ClevrQuestionAnsweringTrainTask(
        cfg.task.clevr_question_answering_train,
        cfg.eval.subset)
    
    dataloader = dataset.get_dataloader(
        batch_size=cfg.eval.batch_size,
        num_workers=8)

    metrics = eval_model(model,dataloader,cfg,cfg.eval.num_eval_samples)
    print('AP:',metrics['AP'])
    print('Accuracy:',metrics['Accuracy'])


if __name__=='__main__':
    main()

