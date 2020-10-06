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
        self.text_fusion_transformer = build_fusion_transformer(cfg.detr)
        self.text_decoder = build_text_decoder(cfg.detr)
        self.answer_head = nn.Linear(cfg.detr.hidden_dim,4)
        self.relevance_embed = nn.Linear(
            cfg.detr.hidden_dim,
            cfg.detr.num_classes + 1)
        self.vision_token = nn.Parameter(
            0.1*torch.randn([cfg.detr.hidden_dim]),requires_grad=True)
        self.lang_token = nn.Parameter(
            0.1*torch.randn([cfg.detr.hidden_dim]),requires_grad=True)
        self.cls_token = nn.Parameter(
            0.1*torch.randn([cfg.detr.hidden_dim]),requires_grad=True)
        
    def forward(self,images,queries):
        outputs = self.detr(images)

        with torch.no_grad():
            query_encodings, token_inputs = self.bert(queries)
        
        query_encodings = self.bert_joiner(query_encodings.detach())

        fused_hs = self.fuse(outputs,query_encodings)

        fused_logits = self.relevance_embed(fused_hs)
        outputs['pred_logits'] = outputs['pred_logits'] + fused_logits[-1]
        if self.cfg.detr.aux_loss:
            for i,aux_outputs in enumerate(outputs['aux_outputs']):
                aux_outputs['pred_logits'] = \
                    aux_outputs['pred_logits'] + fused_logits[i]

        text_fused_hs = self.fuse_text(outputs,query_encodings)

        #import ipdb; ipdb.set_trace()
        memory = torch.cat((fused_hs,text_fused_hs),2)
        L,B,_,D = memory.size()
        target = self.cls_token.view(1,1,1,D).repeat(L,B,1,1)
        outputs['answer_logits'] = self.decode_text(target,memory)
        return outputs

    def fuse_text(self,outputs,query_encodings):
        detr_hs = outputs['detr_hs']
        L = detr_hs.size(0)
        B,T,D = query_encodings.size()
        query_encodings = query_encodings.view(1,B,T,D).repeat(L,1,1,1)
        vision_token = self.vision_token.view(1,1,1,D)
        lang_token = self.lang_token.view(1,1,1,D)
        cls_token = self.cls_token.view(1,1,1,D).repeat(L,B,1,1)
        fused_hs = torch.cat(
            (detr_hs+vision_token,query_encodings+lang_token),2)
        fused_hs = fused_hs.view(L*B,-1,D).permute(1,0,2)
        fused_hs = self.text_fusion_transformer(
            fused_hs[self.cfg.detr.num_queries:],
            fused_hs[:self.cfg.detr.num_queries])
        fused_hs = fused_hs.permute(1,0,2).view(L,B,-1,D)
        return fused_hs

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
        fused_hs = fused_hs.permute(1,0,2).view(L,B,-1,D)
        return fused_hs

    def decode_text(self,target,memory):
        L,B,T,D = memory.size()
        memory = memory.view(L*B,T,D).permute(1,0,2)
        target = target.view(L*B,-1,D).permute(1,0,2)
        #import ipdb; ipdb.set_trace()
        to_decode = self.text_decoder(
            target,memory).permute(1,0,2).view(L,B,-1,D)
        return self.answer_head(to_decode[:,:,0]) # LxBxV


def build_fusion_transformer(cfg):
    decoder_layer = nn.TransformerDecoderLayer(
        d_model=cfg.hidden_dim,
        dropout=cfg.dropout,
        nhead=cfg.nheads)
    
    return nn.TransformerDecoder(decoder_layer,cfg.num_fusion_layers)


build_text_decoder = build_fusion_transformer