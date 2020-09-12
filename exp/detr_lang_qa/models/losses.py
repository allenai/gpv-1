import torch
import torch.nn as nn




class AnswerClassification(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduce=False)

    def forward(self,logits,tgts):
        L,B,C = logits.size()
        logits = logits.permute(1,2,0) # BxCxL
        tgts = tgts.view(B,1).repeat(1,L)
        losses = self.ce_loss(logits,tgts)
        return losses.mean(0).sum()

