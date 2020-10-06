import hydra
import torch
import torch.nn as nn

from utils.set_criterion import SetCriterion
from utils.matcher import HungarianMatcher


class AnswerClassification(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduce=False)

    def forward(self,outputs,targets):
        idxs, filtered_targets = zip(
            *[(i,t) for i,t in enumerate(targets) if 'answer' in t])
        idxs = list(idxs)
        logits = outputs['answer_logits'][:,idxs]
        L,B,C = logits.size()
        logits = logits.permute(1,2,0) # BxCxL
        tgts = torch.stack([t['answer'] for t in filtered_targets])
        tgts = tgts.view(B,1).repeat(1,L)
        losses = self.ce_loss(logits,tgts)
        return {'loss_answer':losses.mean(0).sum()}


class Localization(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.matcher = HungarianMatcher(
            cost_class=cfg.cost_wts.ce,
            cost_bbox=cfg.cost_wts.bbox,
            cost_giou=cfg.cost_wts.giou)

        # wts = cfg.training.loss_wts
        # wts_dict = {
        #     'loss_ce': wts.ce,
        #     'loss_bbox': wts.bbox,
        #     'loss_giou': wts.giou,
        # }

        # if cfg.model.detr.aux_loss:
        #     aux_wts_dict = {}
        #     for i in range(cfg.model.detr.num_decoder_layers - 1):
        #         aux_wts_dict.update({f'{k}_{i}': v for k, v in wts_dict.items()})
            
        #     wts_dict.update(aux_wts_dict)

        self.set_criterion = SetCriterion(
            num_classes=cfg.num_classes,
            matcher=self.matcher,
            weight_dict=None,
            eos_coef=cfg.eos_coef,
            losses=['labels','boxes'])

    def forward(self,outputs,targets):
        idxs, filtered_targets = zip(
            *[(i,t) for i,t in enumerate(targets) if 'boxes' in t])
        idxs = list(idxs)
        filtered_outputs = {
            'pred_logits': outputs['pred_logits'][idxs],
            'pred_boxes': outputs['pred_boxes'][idxs]
        }

        if 'aux_outputs' in outputs:
            aux_outputs = []
            for aux_output in outputs['aux_outputs']:
                aux_outputs.append({
                    'pred_logits': aux_output['pred_logits'][idxs],
                    'pred_boxes': aux_output['pred_boxes'][idxs]
                })
            filtered_outputs['aux_outputs'] = aux_outputs
        
        losses = self.set_criterion(filtered_outputs,filtered_targets)
        to_return = {
            'loss_ce': 0,
            'loss_bbox': 0,
            'loss_giou': 0
        }
        for loss_name in to_return.keys():
            for k,v in losses.items():
                if loss_name in k:
                    to_return[loss_name] += v
        
        return to_return


class GPVCriterion(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg = cfg
        self.criterions = {}
        self.loss_wts = {}
        for loss_module_name,loss_cfg in cfg.items():
            self.criterions[loss_cfg.name] = globals()[loss_module_name](loss_cfg).cuda()
            self.loss_wts.update(loss_cfg.loss_wts)

    def forward(self,outputs,targets):
        loss_dict = {}
        for loss_name,criterion in self.criterions.items():
            losses = criterion(outputs,targets)
            loss_dict.update(losses)
        
        total_loss = 0
        for k,wt in self.loss_wts.items():
            total_loss += wt*loss_dict[k]
        
        return total_loss, loss_dict


@hydra.main(config_path=f'../../../configs',config_name=f"exp/gpv_box_text")
def main(cfg):
    criterion = GPVCriterion(cfg.losses)
    import pdb; pdb.set_trace()


if __name__=='__main__':
    main()