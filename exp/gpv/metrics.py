import os
import h5py
from nltk.tokenize.treebank import TreebankWordDetokenizer
from tqdm import tqdm
import numpy as np
from nltk.tokenize import word_tokenize
import imagesize
import torch

from exp.gpv import evaluators
from data.coco.synonyms import SYNONYMS
import utils.io as io


def vqa_accuracy(model,dataloader,cfg):
    samples = dataloader.dataset.samples
    device = f'cuda:{cfg.gpu}'
    word_to_idx = model.word_to_idx
    idx_to_word = [None]*len(word_to_idx)
    for word,idx in word_to_idx.items():
        idx_to_word[idx] = word

    model.eval()
    
    detokenizer = TreebankWordDetokenizer()
    correct = 0
    total = 0
    end_eval = False
    for data in tqdm(dataloader):
        imgs, queries, targets = data
        imgs = imgs.to(torch.device(device))
        for t in targets:
            for k,v in t.items():
                if not isinstance(v,str):
                    t[k] = v.cuda(device)
        
        answer_tokens,answer_token_ids = model.encode_answers(targets)
        for i,t in enumerate(targets):
            t['answer_token_ids'] = answer_token_ids[i,1:]

        outputs = model(imgs,queries,answer_token_ids=None)

        topk_answers = torch.topk(outputs['answer_logits'][-1],k=1,dim=-1)
        topk_answer_ids = topk_answers.indices.detach().cpu().numpy()
        pred_answers = model.token_ids_to_words(topk_answer_ids[:,:,0])
        B = len(pred_answers)
        for b in range(B):
            if total >= cfg.training.num_val_samples['coco_vqa']:
                end_eval = True
                break
            
            pred_answer = detokenizer.detokenize([w for w in pred_answers[b] if 
                w not in ['__stop__','__pad__']])
            answers = samples[total]['all_answers']
            if pred_answer in answers:
                correctness = min(answers[pred_answer]/3,1)
                correct += correctness
                
            total += 1
        
        if end_eval:
            break

    acc = round(correct / (total + 1e-6),4)
    return acc


def cap_metrics(model,dataloader,cfg):
    samples = dataloader.dataset.samples
    device = f'cuda:{cfg.gpu}'
    word_to_idx = model.word_to_idx
    idx_to_word = [None]*len(word_to_idx)
    for word,idx in word_to_idx.items():
        idx_to_word[idx] = word

    model.eval()
    
    detokenizer = TreebankWordDetokenizer()
    predictions = {}
    total = 0
    end_eval = False
    for data in tqdm(dataloader):
        imgs, queries, targets = data
        imgs = imgs.to(torch.device(device))
        for t in targets:
            for k,v in t.items():
                if not isinstance(v,str):
                    t[k] = v.cuda(device)
        
        answer_tokens,answer_token_ids = model.encode_answers(targets)
        for i,t in enumerate(targets):
            t['answer_token_ids'] = answer_token_ids[i,1:]

        outputs = model(imgs,queries,answer_token_ids=None)

        topk_answers = torch.topk(outputs['answer_logits'][-1],k=1,dim=-1)
        topk_answer_ids = topk_answers.indices.detach().cpu().numpy()
        pred_answers = model.token_ids_to_words(topk_answer_ids[:,:,0])
        B = len(pred_answers)
        for b in range(B):
            if total >= cfg.training.num_val_samples['coco_cap']:
                end_eval = True
                break
            
            pred_answer = detokenizer.detokenize([w for w in pred_answers[b] if 
                w not in ['__stop__','__pad__']])
            cap_id = samples[total]['cap_id']
            predictions[str(cap_id)] = {'answer': pred_answer}
                
            total += 1
        
        if end_eval:
            break

    cap_evaluator = evaluators.CocoCaptioning(samples,predictions,None)
    cap_evaluator.scorers = {
        k:v for k,v in cap_evaluator.scorers.items() if k in ['Bleu','Cider']}
    metrics = cap_evaluator.evaluate()
    return metrics['scores']


def create_coco_vocab_mask(model,use_syns=True):
    L = len(model.vocab)
    mask = -10000*np.ones([L],dtype=np.float32)
    tokens = []
    # SYNONYMS is a dict whose keys are coco categories
    for coco_cls in SYNONYMS:
        syns = [coco_cls]
        if use_syns is True:
            syns = SYNONYMS[coco_cls]
        
        for syn in syns:
            for token in word_tokenize(syn):
                if token in model.word_to_idx:
                    idx = model.word_to_idx[token]
                    mask[idx] = 0
                    tokens.append(token)
    
    for token in ['__stop__','__pad__']:
        idx = model.word_to_idx[token]
        mask[idx] = 0
        tokens.append(token)

    return tokens, mask


def cls_metrics(model,dataloader,cfg):
    samples = dataloader.dataset.samples
    device = f'cuda:{cfg.gpu}'
    word_to_idx = model.word_to_idx
    idx_to_word = [None]*len(word_to_idx)
    for word,idx in word_to_idx.items():
        idx_to_word[idx] = word

    tokens,vocab_mask = create_coco_vocab_mask(model)
    vocab_mask = torch.FloatTensor(vocab_mask).cuda(cfg.gpu)

    model.eval()
    
    detokenizer = TreebankWordDetokenizer()
    predictions = {}
    total = 0
    end_eval = False
    for data in tqdm(dataloader):
        imgs, queries, targets = data
        imgs = imgs.to(torch.device(device))
        for t in targets:
            for k,v in t.items():
                if not isinstance(v,str):
                    t[k] = v.cuda(device)
        
        answer_tokens,answer_token_ids = model.encode_answers(targets)
        for i,t in enumerate(targets):
            t['answer_token_ids'] = answer_token_ids[i,1:]

        outputs = model(imgs,queries,answer_token_ids=None,vocab_mask=vocab_mask)

        topk_answers = torch.topk(outputs['answer_logits'][-1],k=1,dim=-1)
        topk_answer_ids = topk_answers.indices.detach().cpu().numpy()
        pred_answers = model.token_ids_to_words(topk_answer_ids[:,:,0])
        B = len(pred_answers)
        for b in range(B):
            if total >= cfg.training.num_val_samples['coco_cls']:
                end_eval = True
                break
            
            pred_answer = detokenizer.detokenize([w for w in pred_answers[b] if 
                w not in ['__stop__','__pad__']])
            sample_id = samples[total]['id']
            predictions[str(sample_id)] = {'answer': pred_answer}
                
            total += 1
        
        if end_eval:
            break

    cls_evaluator = evaluators.CocoClassification(samples,predictions,None)
    metrics = cls_evaluator.evaluate()
    return metrics['accuracy']['all']


def update_samples_with_image_size(image_dir,samples):
    for sample in tqdm(samples):
        image_id = sample['image']['image_id']
        image_subset = sample['image']['subset']
        image_filename = os.path.join(
            os.path.join(image_dir,image_subset),
            'COCO_'+image_subset+'_'+str(image_id).zfill(12)+'.jpg')

        W,H = imagesize.get(image_filename)
        sample['image']['W'] = W
        sample['image']['H'] = H
    
    return samples


def det_metrics(model,dataloader,cfg):
    samples = dataloader.dataset.samples
    device = f'cuda:{cfg.gpu}'
    word_to_idx = model.word_to_idx
    idx_to_word = [None]*len(word_to_idx)
    for word,idx in word_to_idx.items():
        idx_to_word[idx] = word

    model.eval()
    
    detokenizer = TreebankWordDetokenizer()
    predictions = {}
    total = 0
    end_eval = False
    eval_dir = os.path.join(cfg.exp_dir,'train_time_eval')
    io.mkdir_if_not_exists(eval_dir)
    boxes_h5py_path = os.path.join(
        eval_dir,f'det_{dataloader.dataset.subset}_boxes.h5py')
    boxes_h5py = h5py.File(boxes_h5py_path,'w')
    for data in tqdm(dataloader):
        imgs, queries, targets = data
        imgs = imgs.to(torch.device(device))
        for t in targets:
            for k,v in t.items():
                if not isinstance(v,str):
                    t[k] = v.cuda(device)
        
        answer_tokens,answer_token_ids = model.encode_answers(targets)
        for i,t in enumerate(targets):
            t['answer_token_ids'] = answer_token_ids[i,1:]

        outputs = model(imgs,queries,answer_token_ids=None)
        relevance = outputs['pred_relevance_logits'].softmax(-1).detach().cpu().numpy()
        pred_boxes = outputs['pred_boxes'].detach().cpu().numpy()
        B = len(targets)
        for b in range(B):
            if total >= cfg.training.num_val_samples['coco_det']:
                end_eval = True
                break
            
            scores, boxes = zip(*sorted(zip(
                relevance[b,:,0].tolist(),pred_boxes[b].tolist()),
                key=lambda x: x[0],reverse=True))
            scores = np.array(scores,dtype=np.float32)
            boxes = np.array(boxes,dtype=np.float32)
            sample_id = samples[total]['id']
            grp = boxes_h5py.create_group(str(sample_id))
            grp.create_dataset('boxes',data=boxes)
            grp.create_dataset('relevance',data=scores)
            predictions[str(sample_id)] = {
                'answer': ''
            }
                
            total += 1
        
        if end_eval:
            break

    boxes_h5py.close()

    samples = update_samples_with_image_size(
        cfg.task_configs.image_dir,
        samples)

    boxes_h5py = h5py.File(boxes_h5py_path,'r')
    det_evaluator = evaluators.CocoDetection(samples,predictions,boxes_h5py)
    metrics = det_evaluator.evaluate()
    boxes_h5py.close()
    os.remove(boxes_h5py_path)
    return metrics['mAP']
    # APs = list(metrics['AP'].values())
    # print('Num class APs:',len(APs))
    # APs = [a for a in APs if not np.isnan(a)]
    # print('Num non-nan class APs:',len(APs))
    # return np.mean(APs)


def refexp_metrics(model,dataloader,cfg):
    samples = dataloader.dataset.samples
    device = f'cuda:{cfg.gpu}'
    word_to_idx = model.word_to_idx
    idx_to_word = [None]*len(word_to_idx)
    for word,idx in word_to_idx.items():
        idx_to_word[idx] = word

    model.eval()
    
    detokenizer = TreebankWordDetokenizer()
    predictions = {}
    total = 0
    end_eval = False
    eval_dir = os.path.join(cfg.exp_dir,'train_time_eval')
    io.mkdir_if_not_exists(eval_dir)
    boxes_h5py_path = os.path.join(
        eval_dir,f'det_{dataloader.dataset.subset}_boxes.h5py')
    boxes_h5py = h5py.File(boxes_h5py_path,'w')
    for data in tqdm(dataloader):
        imgs, queries, targets = data
        imgs = imgs.to(torch.device(device))
        for t in targets:
            for k,v in t.items():
                if not isinstance(v,str):
                    t[k] = v.cuda(device)
        
        answer_tokens,answer_token_ids = model.encode_answers(targets)
        for i,t in enumerate(targets):
            t['answer_token_ids'] = answer_token_ids[i,1:]

        outputs = model(imgs,queries,answer_token_ids=None)
        relevance = outputs['pred_relevance_logits'].softmax(-1).detach().cpu().numpy()
        pred_boxes = outputs['pred_boxes'].detach().cpu().numpy()
        B = len(targets)
        for b in range(B):
            if total >= cfg.training.num_val_samples['refcocop']:
                end_eval = True
                break
            
            scores, boxes = zip(*sorted(zip(
                relevance[b,:,0].tolist(),pred_boxes[b].tolist()),
                key=lambda x: x[0],reverse=True))
            scores = np.array(scores,dtype=np.float32)
            boxes = np.array(boxes,dtype=np.float32)
            sample_id = samples[total]['sent_id']
            grp = boxes_h5py.create_group(str(sample_id))
            grp.create_dataset('boxes',data=boxes)
            grp.create_dataset('relevance',data=scores)
            predictions[str(sample_id)] = {
                'answer': ''
            }
                
            total += 1
        
        if end_eval:
            break

    boxes_h5py.close()

    samples = update_samples_with_image_size(
        cfg.task_configs.image_dir,
        samples)

    boxes_h5py = h5py.File(boxes_h5py_path,'r')
    refexp_evaluator = evaluators.RefCocop(samples,predictions,boxes_h5py)
    metrics = refexp_evaluator.evaluate()
    boxes_h5py.close()
    os.remove(boxes_h5py_path)
    return metrics['mAP']