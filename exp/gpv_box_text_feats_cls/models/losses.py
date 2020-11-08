import hydra
import torch
import torch.nn as nn

from utils.set_criterion import SetCriterion
from utils.matcher import HungarianMatcher


class AnswerClassification(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        ignore_index = -100
        if cfg.pad_idx is not None:
            print('Ignoring pad idx:',cfg.pad_idx)
            ignore_index = cfg.pad_idx

        self.ce_loss = nn.CrossEntropyLoss(
            reduce=False,ignore_index=ignore_index)

    def forward(self,outputs,targets):
        idx_filtered_targets = [(
            i,t) for i,t in enumerate(targets) if 'answer' in t]
        if len(idx_filtered_targets)==0:
            return {'loss_answer':None}

        idxs, filtered_targets = zip(*idx_filtered_targets)
        idxs = list(idxs)
        logits = outputs['answer_logits'][:,idxs]
        L,B,S,V = logits.size()
        logits = logits.permute(1,3,2,0) # BxVxSxL
        tgts = torch.stack([t['answer_token_ids'] for t in filtered_targets])
        tgts = tgts.view(B,S,1).repeat(1,1,L)
        losses = self.ce_loss(logits,tgts) # BxSXL
        return {'loss_answer':losses.mean(0).sum(0).sum()}


class Localization(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.matcher = HungarianMatcher(
            cost_class=cfg.cost_wts.ce,
            cost_bbox=cfg.cost_wts.bbox,
            cost_giou=cfg.cost_wts.giou)

        self.set_criterion = SetCriterion(
            num_classes=cfg.num_classes,
            matcher=self.matcher,
            weight_dict=None,
            eos_coef=cfg.eos_coef,
            losses=['labels','boxes'])

    def forward(self,outputs,targets):
        idx_filtered_targets = [
            (i,t) for i,t in enumerate(targets) if 'boxes' in t]
        if len(idx_filtered_targets)==0:
            return {
                'loss_ce': None,
                'loss_bbox': None,
                'loss_giou': None
            }
        
        idxs, filtered_targets = zip(*idx_filtered_targets)
        idxs = list(idxs)
        filtered_outputs = {
            'pred_relevance_logits': outputs['pred_relevance_logits'][idxs],
            'pred_boxes': outputs['pred_boxes'][idxs]
        }

        if 'aux_outputs' in outputs:
            aux_outputs = []
            for aux_output in outputs['aux_outputs']:
                aux_outputs.append({
                    'pred_relevance_logits': aux_output['pred_relevance_logits'][idxs],
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
        self.criterion_names = []
        self.loss_wts = {}
        for loss_module_name,loss_cfg in cfg.items():
            setattr(
                self,
                loss_cfg.name,
                globals()[loss_module_name](loss_cfg))
            self.criterion_names.append(loss_cfg.name)
            self.loss_wts.update(loss_cfg.loss_wts)

    def forward(self,outputs,targets):
        loss_dict = {}
        for name in self.criterion_names:
            criterion = getattr(self,name)
            losses = criterion(outputs,targets)
            loss_dict.update(losses)
        
        all_losses_none = True
        for k,v in loss_dict.items():
            if v is not None:
                all_losses_none = False
                break
        
        if all_losses_none is True:
            return None, loss_dict

        total_loss = 0
        for k,wt in self.loss_wts.items():
            if loss_dict[k] is not None:
                total_loss += wt*loss_dict[k]
        
        return total_loss, loss_dict


@hydra.main(config_path=f'../../../configs',config_name=f"exp/gpv_box_text")
def main(cfg):
    criterion = GPVCriterion(cfg.losses)
    import pdb; pdb.set_trace()


if __name__=='__main__':
    main()