import nltk
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

from .detr import create_detr
from .bert import Bert
from .answer_head import build_answer_head
import utils.io as io

nltk.download('punkt')


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
        self.answer_head = build_answer_head(cfg,self.bert_joiner)
        self.vocab = self.answer_head.vocab
        #self.vocab = io.load_json_object(cfg.vocab)
        self.word_to_idx = {w:i for i,w in enumerate(self.vocab)}
        #self.answer_head = nn.Linear(cfg.detr.hidden_dim,len(self.vocab))
        self.answer_input_embedings = nn.Embedding(
            len(self.vocab),
            cfg.detr.hidden_dim)
        self.relevance_embed = nn.Linear(
            cfg.detr.hidden_dim,
            cfg.detr.num_classes + 1)
        self.vision_token = nn.Parameter(
            0.1*torch.randn([cfg.detr.hidden_dim]),requires_grad=True)
        self.lang_token = nn.Parameter(
            0.1*torch.randn([cfg.detr.hidden_dim]),requires_grad=True)
        self.relevance_tokens = nn.Parameter(
            0.1*torch.randn([2,cfg.detr.hidden_dim]),requires_grad=True)
        # self.cls_token = nn.Parameter(
        #     0.1*torch.randn([cfg.detr.hidden_dim]),requires_grad=True)
        
    def forward(self,images,queries,answer_token_ids):
        outputs = self.detr(images)

        with torch.no_grad():
            query_encodings, token_inputs = self.bert(queries)
        
        query_encodings = self.bert_joiner(query_encodings.detach())

        fused_hs = self.fuse(outputs,query_encodings) # LxBxRxD

        fused_logits = self.relevance_embed(fused_hs)
        outputs['pred_logits'] = outputs['pred_logits'] + fused_logits[-1] #BxRx2
        if self.cfg.detr.aux_loss:
            for i,aux_outputs in enumerate(outputs['aux_outputs']):
                aux_outputs['pred_logits'] = \
                    aux_outputs['pred_logits'] + fused_logits[i]

        text_fused_hs = self.fuse_text(outputs,query_encodings)

        fused_hs = self.condition_on_relevance(outputs['pred_logits'],fused_hs)
        memory = torch.cat((fused_hs,text_fused_hs),2)
        L,B,_,D = memory.size()

        #target = self.cls_token.view(1,1,1,D).repeat(L,B,1,1)
        
        if answer_token_ids is None:
            cls_token_id = torch.LongTensor(
                [self.word_to_idx['__cls__']]).cuda()
            target_token_ids = cls_token_id.view(1,1,1).repeat(L,B,1)
            for t in range(self.cfg.max_text_len-1):
                target = self.answer_input_embedings(target_token_ids)
                #target = target.view(1,*target.size()).repeat(L,1,1,1)
                answer_logits = self.decode_text(target,memory) # LxBxSxV
                top_ids = torch.topk(answer_logits,k=1,dim=-1).indices[:,:,-1] # LxBx1
                target_token_ids = torch.cat((target_token_ids,top_ids),-1)
            
            target = self.answer_input_embedings(target_token_ids) # BxTXD
            #target = target.view(1,*target.size()).repeat(L,1,1,1)
            outputs['answer_logits'] = self.decode_text(target,memory)

        else:
            target = self.answer_input_embedings(answer_token_ids) # BxTXD
            target = target.view(1,*target.size()).repeat(L,1,1,1)
            outputs['answer_logits'] = self.decode_text(target,memory)[:,:,:-1]

        return outputs

    def condition_on_relevance(self,relevance_logits,fused_hs):
        L,B,R,D = fused_hs.size()
        prob = relevance_logits.softmax(-1) # BxRx2
        prob = prob.view(B,R,2,1)
        relevance_tokens = self.relevance_tokens.view(1,1,2,D)
        relevance_tokens = torch.sum(relevance_tokens*prob,2) # BxRxD
        relevance_tokens = relevance_tokens.view(1,B,R,D)
        fused_hs = fused_hs + relevance_tokens
        return fused_hs

    def encode_answers(self,targets):
        B = len(targets)
        answers = ['']*B
        for i,t in enumerate(targets):
            if 'answer' in t:
                answers[i] = t['answer']

        padded_inputs = [None]*len(answers)
        S = 0
        for i,answer in enumerate(answers):
            if answer=='':
                sent = f'__cls__ __stop__'
            else:
                sent = f'__cls__ {answer} __stop__'
            padded_inputs[i] = [w.lower() for w in sent.split(' ')]
            S = max(S,len(padded_inputs[i]))
        
        padded_token_ids = [None]*len(answers)
        for i,padded_tokens in enumerate(padded_inputs):
            padded_tokens.extend(['__pad__']*(S-len(padded_tokens)))
            token_ids = [None]*S
            for j in range(S):
                token_ids[j] = self.word_to_idx[padded_tokens[j]]

            padded_token_ids[i] = token_ids

        padded_token_ids = torch.LongTensor(padded_token_ids).cuda()
        return padded_inputs, padded_token_ids
        
    def token_ids_to_words(self,token_ids):
        B,S = token_ids.shape
        #token_ids = token_ids.cpu().numpy()
        words = [None]*B
        for i in range(B):
            words[i] = [None]*S
            for j in range(S):
                words[i][j] = self.vocab[token_ids[i,j]]
        
        return words
        
    @property
    def cls_token(self):
        return self.answer_input_embedings(
            torch.LongTensor([self.word_to_idx['__cls__']]).cuda())[0]

    def fuse_text(self,outputs,query_encodings):
        detr_hs = outputs['detr_hs']
        detr_hs = self.condition_on_relevance(outputs['pred_logits'],detr_hs)
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
        L,B,Tm,D = memory.size()
        _,_,Tt,D = target.size()
        memory = memory.view(L*B,Tm,D).permute(1,0,2)
        target = target.view(L*B,Tt,D).permute(1,0,2)
        #import ipdb; ipdb.set_trace()
        tgt_mask = torch.zeros((Tt,Tt))
        for t in range(Tt):
            for j in range(t+1,Tt):
                tgt_mask[t,j] = 1
        
        tgt_mask = tgt_mask.byte().cuda()#.view(T,T).repeat(12,1,1)
        to_decode = self.text_decoder(
            target,memory,tgt_mask).permute(1,0,2).view(L,B,-1,D)
        return self.answer_head(to_decode)
        #return self.answer_head(to_decode[:,:,:-1]) # LxBxSxV


def build_fusion_transformer(cfg):
    decoder_layer = nn.TransformerDecoderLayer(
        d_model=cfg.hidden_dim,
        dropout=cfg.dropout,
        nhead=cfg.nheads)
    
    return nn.TransformerDecoder(decoder_layer,cfg.num_fusion_layers)


build_text_decoder = build_fusion_transformer