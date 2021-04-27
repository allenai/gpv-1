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
    for img_path, query in inputs:
        img,_ = read_image(img_path,resize_image=True)
        proc_img = (255*img).astype(np.uint8)
        proc_img = transforms(proc_img).cuda()
        proc_inputs.append((proc_img,query))
    
    return collate_fn(proc_inputs)


def decode_outputs(outputs):
    detokenizer = TreebankWordDetokenizer()
    relevance = outputs['pred_relevance_logits'].softmax(-1).detach().cpu().numpy()
    pred_boxes = outputs['pred_boxes'].detach().cpu().numpy()
    B = pred_boxes.shape[0]
    
    decoded_outputs = []
    for b in range(B):
        scores, boxes = zip(*sorted(zip(
            relevance[b,:,0].tolist(),pred_boxes[b].tolist()),
            key=lambda x: x[0],reverse=True))
        scores = np.array(scores,dtype=np.float32)
        boxes = np.array(boxes,dtype=np.float32)
        answers = [detokenizer.detokenize(ans) for ans in outputs['answers'][b]]
        decoded_outputs.append({
            'answers': answers,
            'answer_probs': outputs['answer_probs'][b],
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

    inputs = [(cfg.inputs.img,cfg.inputs.query)]

    images, queries = preprocess(inputs,transforms)
    outputs = model.forward_beam_search(images,queries,beam_size=5)
    prediction = decode_outputs(outputs)[0]

    prediction['boxes'] = prediction['boxes'][:cfg.num_output_boxes]
    prediction['relevance'] = prediction['relevance'][:cfg.num_output_boxes]
    print(prediction)

if __name__=='__main__':
    main()