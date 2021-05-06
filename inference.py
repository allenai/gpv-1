import hydra
from hydra.experimental import compose, initialize
from omegaconf import OmegaConf
import torch
import torchvision.transforms as T
import numpy as np
from nltk.tokenize.treebank import TreebankWordDetokenizer

from exp.gpv.models.gpv import GPV
from utils.detr_misc import collate_fn
from inference_util import *


def preprocess(inputs,transforms):
    proc_inputs = []
    for img, query in inputs:
        proc_img = (255*img).astype(np.uint8)
        proc_img = transforms(proc_img).cuda()
        proc_inputs.append((proc_img,query))
    
    return collate_fn(proc_inputs)


def decode_outputs(outputs,model):
    detokenizer = TreebankWordDetokenizer()
    relevance = outputs['pred_relevance_logits'].softmax(-1).detach().cpu().numpy()
    pred_boxes = outputs['pred_boxes'].detach().cpu().numpy()
    topk_answers = torch.topk(outputs['answer_logits'][-1],k=1,dim=-1)
    topk_answer_ids = topk_answers.indices.detach().cpu().numpy()
    pred_answers = model.token_ids_to_words(topk_answer_ids[:,:,0])
    decoded_outputs = []
    for b in range(len(pred_answers)):
        scores, boxes = zip(*sorted(zip(
            relevance[b,:,0].tolist(),pred_boxes[b].tolist()),
            key=lambda x: x[0],reverse=True))
        scores = np.array(scores,dtype=np.float32)
        boxes = np.array(boxes,dtype=np.float32)
        answer = []
        for token in pred_answers[b]:
            if token in ['__stop__','__pad__']:
                break
            answer.append(token)
        answer = detokenizer.detokenize(answer)
        decoded_outputs.append({
            'answer': answer,
            'boxes': boxes,
            'relevance': scores})
    
    return decoded_outputs


@hydra.main(config_path=f'configs',config_name=f"exp/gpv_inference_cmdline")
def main(cfg):
    print(cfg.inputs)

    model = GPV(cfg.model).cuda().eval()
    loaded_dict = torch.load(cfg.ckpt, map_location='cuda:0')['model']
    state_dict = model.state_dict()
    for k,v in state_dict.items():
        state_dict[k] = loaded_dict[f'module.{k}']
        state_dict[k].requires_grad = False
    model.load_state_dict(state_dict)

    transforms = T.Compose([
        T.ToPILImage(mode='RGB'),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    img,_ = read_image(cfg.inputs.img,resize_image=False)
    #imshow((255*img[:,:,::-1]).astype(np.uint8)) # scale pixel values, RGB to BGR (because imshow uses opencv), and convert to uint8

    inputs = [(img,cfg.inputs.query)]

    images, queries = preprocess(inputs,transforms)
    outputs = model(images,queries,None)
    prediction = decode_outputs(outputs,model)[0]

    prediction['boxes'] = prediction['boxes'][:cfg.num_output_boxes]
    prediction['relevance'] = prediction['relevance'][:cfg.num_output_boxes]
    
    for pred_type, pred_value in prediction.items():
        print('-'*80)
        print(pred_type)
        print('-'*80)
        print(pred_value)

if __name__=='__main__':
    main()