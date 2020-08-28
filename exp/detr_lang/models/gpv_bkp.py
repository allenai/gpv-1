import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

from .detr_lang import create_detr_lang
from .bert import Bert

class GPV(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.detr = create_detr_lang(cfg.detr)
        self.bert = Bert()
        self.bert_joiner = nn.Linear(768,cfg.detr.hidden_dim)
        
    def forward(self,images,queries):
        with torch.no_grad():
            query_encodings, token_inputs = self.bert(queries)
        
        query_encodings = self.bert_joiner(query_encodings.detach())
        return self.detr(images,query_encodings)


