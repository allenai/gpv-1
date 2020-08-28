import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

from .detr import create_detr
from .bert import Bert

class GPV(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg = cfg
        self.detr = create_detr(cfg.detr)
        self.bert = Bert()
        self.bert_joiner = nn.Linear(768,cfg.detr.hidden_dim)
        self.fusion_transformer = build_fusion_transformer(cfg.detr)
        self.relevance_embed = nn.Linear(
            cfg.detr.hidden_dim,
            cfg.detr.num_classes + 1)
        self.vision_token = nn.Parameter(
            0.1*torch.randn([cfg.detr.hidden_dim]),requires_grad=True)
        self.lang_token = nn.Parameter(
            0.1*torch.randn([cfg.detr.hidden_dim]),requires_grad=True)
        
    def forward(self,images,queries):
        outputs = self.detr(images)

        with torch.no_grad():
            query_encodings, token_inputs = self.bert(queries)
        
        query_encodings = self.bert_joiner(query_encodings.detach())

        fused_hs = self.fuse(outputs,query_encodings)
        #import ipdb; ipdb.set_trace()

        fused_logits = self.relevance_embed(fused_hs)
        outputs['pred_logits'] = outputs['pred_logits'] + fused_logits[-1]
        if self.cfg.detr.aux_loss:
            for i,aux_outputs in enumerate(outputs['aux_outputs']):
                aux_outputs['pred_logits'] = \
                    aux_outputs['pred_logits'] + fused_logits[i]
        
        return outputs

    def fuse(self,outputs,query_encodings):
        detr_hs = outputs['detr_hs']
        L = detr_hs.size(0)
        B,T,D = query_encodings.size()
        query_encodings = query_encodings.view(1,B,T,D).repeat(L,1,1,1)
        vision_token = self.vision_token.view(1,1,1,D)
        lang_token = self.lang_token.view(1,1,1,D)
        fused_hs = torch.cat(
            (detr_hs+vision_token,query_encodings+lang_token),2)
        fused_hs = fused_hs.view(L*B,-1,D).permute(1,0,2)
        fused_hs = self.fusion_transformer(
            fused_hs[:self.cfg.detr.num_queries],
            fused_hs[self.cfg.detr.num_queries:])
        # fused_hs = self.fusion_transformer(fused_hs)
        fused_hs = fused_hs.permute(1,0,2).view(L,B,-1,D)
        fused_hs = fused_hs[:,:,:self.cfg.detr.num_queries]
        fused_hs = fused_hs #+pretext_hs
        return fused_hs

def build_fusion_transformer(cfg):
    # encoder_layer = nn.TransformerEncoderLayer(
    #     d_model=cfg.hidden_dim,
    #     dropout=cfg.dropout,
    #     nhead=cfg.nheads)
    
    # return nn.TransformerEncoder(encoder_layer,cfg.num_fusion_layers)

    decoder_layer = nn.TransformerDecoderLayer(
        d_model=cfg.hidden_dim,
        dropout=cfg.dropout,
        nhead=cfg.nheads)
    
    return nn.TransformerDecoder(decoder_layer,cfg.num_fusion_layers)


