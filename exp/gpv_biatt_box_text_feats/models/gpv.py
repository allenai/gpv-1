import copy
import nltk
from nltk.tokenize import word_tokenize
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

from .detr import create_detr
from .bert import Bert
from .answer_head import build_answer_head
from .vilbert import BertConnectionLayer
from .losses import GPVCriterion
import utils.io as io


def build_transformer_decoder(cfg):
    decoder_layer = nn.TransformerDecoderLayer(
        d_model=cfg.hidden_dim,
        dropout=cfg.dropout,
        nhead=cfg.nheads)
    
    return nn.TransformerDecoder(decoder_layer,cfg.num_layers)


class AnswerInputEmbedding(nn.Module):
    def __init__(self,weight,transform,freeze_embeddings):
        super().__init__()
        self.transform = transform
        self.embedding_layer = nn.Embedding.from_pretrained(
            weight,freeze=freeze_embeddings)
    
    def forward(self,token_ids):
        embed = self.embedding_layer(token_ids)
        return self.transform(embed)
        

class GPV(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg = cfg

        # visual stream
        self.detr = create_detr(cfg.detr)
        self.init_detr_params = []

        # text stream
        self.bert = Bert()
        self.bert_joiner = nn.Linear(
            cfg.bert_joiner.bert_dim,
            cfg.bert_joiner.out_dim)
        
        # encode vision with language context and vice versa
        #self.vl_transformer = build_transformer_decoder(cfg.vl_transformer)
        #self.lv_transformer = build_transformer_decoder(cfg.lv_transformer)
        layer = BertConnectionLayer(cfg.co_att)
        self.co_att_transformer = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(cfg.co_att.num_layers)])
            
        # relevance predictor that operates on vl output
        self.relevance_predictor = nn.Linear(
            cfg.detr.hidden_dim,
            cfg.detr.num_classes + 1)

        # text decoder
        self.text_decoder = build_transformer_decoder(cfg.text_decoder)
        answer_transform = nn.Linear(
            cfg.bert_joiner.bert_dim,
            cfg.bert_joiner.out_dim)
        self.answer_head = build_answer_head(cfg,answer_transform)
        self.vocab = self.answer_head.vocab
        self.word_to_idx = {w:i for i,w in enumerate(self.vocab)}
        answer_input_transform = nn.Linear(
            cfg.bert_joiner.bert_dim,
            cfg.bert_joiner.out_dim)
        self.answer_input_embedings = AnswerInputEmbedding(
            self.answer_head.vocab_embed.data,
            answer_input_transform,
            freeze_embeddings=(self.cfg.answer_head!='linear'))
    
        # indicator tokens
        self.vision_token = nn.Parameter(
            0.1*torch.randn([cfg.detr.hidden_dim]),requires_grad=True)
        self.lang_token = nn.Parameter(
            0.1*torch.randn([cfg.detr.hidden_dim]),requires_grad=True)
        self.relevance_tokens = nn.Parameter(
            0.1*torch.randn([2,cfg.detr.hidden_dim]),requires_grad=True)

        self.criterion = GPVCriterion(self.cfg.losses)

    
    def load_pretr_detr(self):
        loaded_model = torch.load(self.cfg.pretr_detr)['model'] # eg. key backbone.0.body.layer2.1.conv1.weight
        curr_model = self.state_dict()
        #init_detr_params = []
        for lk in loaded_model.keys():
            detr_lk ='detr.'+lk
            if detr_lk in curr_model:
                #print(detr_lk)
                if curr_model[detr_lk].size()==loaded_model[lk].size():
                    self.init_detr_params.append(detr_lk)
                    curr_model[detr_lk] = loaded_model[lk]
                else:
                    print(f'    {lk} size does not match')
        
        self.load_state_dict(curr_model)

    def forward(self,feats,queries,answer_token_ids,targets=None,vocab_mask=None):
        device = self.vision_token.device
        outputs = self.detr(feats)

        with torch.no_grad():
            query_encodings, token_inputs = self.bert(queries,device)
        
        query_encodings = self.bert_joiner(query_encodings.detach())

        #import ipdb; ipdb.set_trace()
        lv_hs = query_encodings
        vl_hs = outputs['detr_hs'][-1]
        for layer in self.co_att_transformer:
            lv_hs, vl_hs, _ = layer(
                input_tensor1=lv_hs,
                attention_mask1=None,
                input_tensor2=vl_hs,
                attention_mask2=None)
        
        B,Tl,D = lv_hs.size()
        _,Tv,_ = vl_hs.size()
        lv_hs = lv_hs.view(1,B,Tl,D)
        vl_hs = vl_hs.view(1,B,Tv,D)
        #import ipdb; ipdb.set_trace()
        # vl encoding and relevance prediction
        #vl_hs = self.encode_v_with_l_context(outputs,query_encodings) # LxBxRxD
        relevance_logits = self.relevance_predictor(vl_hs)
        outputs['pred_relevance_logits'] = \
            outputs['pred_relevance_logits'] + relevance_logits[-1] #BxRx2
        if self.cfg.detr.aux_loss:
            for i,aux_outputs in enumerate(outputs['aux_outputs']):
                aux_outputs['pred_relevance_logits'] = \
                    aux_outputs['pred_relevance_logits'] + relevance_logits[i]
        
        # condition vl encoding on relevance prediction
        # vl_hs = self.condition_on_relevance(
        #     outputs['pred_relevance_logits'],vl_hs)

        # lv encoding
        #lv_hs = self.encode_l_with_v_context(outputs,query_encodings)

        # concat vl and lv to create a memory for text decoding
        memory = torch.cat((vl_hs,lv_hs),2)
        L,B,_,D = memory.size()
        
        if answer_token_ids is None:
            # sample text without teacher forcing
            cls_token_id = torch.LongTensor([self.word_to_idx['__cls__']]).cuda(device)
            target_token_ids = cls_token_id.view(1,1,1).repeat(L,B,1)
            for t in range(self.cfg.max_text_len-1):
                target = self.answer_input_embedings(target_token_ids)
                answer_logits = self.decode_text(target,memory) # LxBxSxV
                answer_logits = answer_logits[:,:,-1]
                if vocab_mask is not None:
                    answer_logits = answer_logits + vocab_mask
                top_ids = torch.topk(answer_logits,k=1,dim=-1).indices#[:,:,-1] # LxBx1
                target_token_ids = torch.cat((target_token_ids,top_ids),-1)
            
            target = self.answer_input_embedings(target_token_ids) # BxTXD
            answer_logits = self.decode_text(target,memory)
            if vocab_mask is not None:
                answer_logits = answer_logits + vocab_mask

            outputs['answer_logits'] = answer_logits
        else:
            # sample text with teacher forcing
            target = self.answer_input_embedings(answer_token_ids) # BxTXD
            target = target.view(1,*target.size()).repeat(L,1,1,1)
            outputs['answer_logits'] = self.decode_text(target,memory)[:,:,:-1]

        if targets is None:
            return outputs
        else:
            total_loss = self.criterion(outputs,targets)[0]
            return total_loss

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

        if self.cfg.answering_type=='classification':
            padded_inputs = [None]*len(answers)
            padded_token_ids = [None]*len(answers)
            for i,answer in enumerate(answers):
                padded_inputs[i] = ['__cls__',answer]
                padded_token_ids[i] = []
                for token in padded_inputs[i]:
                    if token in self.word_to_idx:
                        token_id = self.word_to_idx[token]
                    else:
                        token_id = self.word_to_idx['__unk__']
                    
                    padded_token_ids[i].append(token_id)

            device = self.vision_token.device
            padded_token_ids = torch.LongTensor(padded_token_ids).cuda(device)
        
        elif self.cfg.answering_type=='generation':
            padded_inputs = [None]*len(answers)
            S = 0
            for i,answer in enumerate(answers):
                if answer=='':
                    sent = f'__cls__ __stop__'
                else:
                    sent = f'__cls__ {answer} __stop__'
                padded_inputs[i] = [w.lower() for w in word_tokenize(sent)]
                S = max(S,len(padded_inputs[i]))
            
            padded_token_ids = [None]*len(answers)
            for i,padded_tokens in enumerate(padded_inputs):
                padded_tokens.extend(['__pad__']*(S-len(padded_tokens)))
                token_ids = [None]*S
                for j in range(S):
                    if padded_tokens[j] in self.word_to_idx:
                        token_ids[j] = self.word_to_idx[padded_tokens[j]]
                    else:
                        token_ids[j] = self.word_to_idx['__unk__']

                padded_token_ids[i] = token_ids[:self.cfg.max_text_len]

            device = self.vision_token.device
            padded_token_ids = torch.LongTensor(padded_token_ids).cuda(device)
        
        else:
            raise NotImplementedError
        
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
        device = self.vision_token.device
        return self.answer_input_embedings(
            torch.LongTensor([self.word_to_idx['__cls__']]).cuda(device))[0]

    def encode_l_with_v_context(self,outputs,query_encodings):
        detr_hs = outputs['detr_hs']
        # detr_hs = self.condition_on_relevance(
        #     outputs['pred_relevance_logits'],detr_hs)
        L = detr_hs.size(0)
        B,T,D = query_encodings.size()
        query_encodings = query_encodings.view(1,B,T,D).repeat(L,1,1,1)
        vision_token = self.vision_token.view(1,1,1,D)
        lang_token = self.lang_token.view(1,1,1,D)
        cls_token = self.cls_token.view(1,1,1,D).repeat(L,B,1,1)
        fused_hs = torch.cat(
            (detr_hs+vision_token,query_encodings+lang_token),2)
        fused_hs = fused_hs.view(L*B,-1,D).permute(1,0,2)
        fused_hs = self.lv_transformer(
            fused_hs[self.cfg.detr.num_queries:],
            fused_hs[:self.cfg.detr.num_queries])
        fused_hs = fused_hs.permute(1,0,2).view(L,B,-1,D)
        return fused_hs

    def encode_v_with_l_context(self,outputs,query_encodings):
        detr_hs = outputs['detr_hs']
        L = detr_hs.size(0)
        B,T,D = query_encodings.size()
        query_encodings = query_encodings.view(1,B,T,D).repeat(L,1,1,1)
        vision_token = self.vision_token.view(1,1,1,D)
        lang_token = self.lang_token.view(1,1,1,D)
        fused_hs = torch.cat(
            (detr_hs+vision_token,query_encodings+lang_token),2)
        fused_hs = fused_hs.view(L*B,-1,D).permute(1,0,2)
        fused_hs = self.vl_transformer(
            fused_hs[:self.cfg.detr.num_queries],
            fused_hs[self.cfg.detr.num_queries:])
        fused_hs = fused_hs.permute(1,0,2).view(L,B,-1,D)
        return fused_hs

    def decode_text(self,target,memory):
        L,B,Tm,D = memory.size()
        _,_,Tt,D = target.size()
        memory = memory.view(L*B,Tm,D).permute(1,0,2)
        target = target.view(L*B,Tt,D).permute(1,0,2)
        tgt_mask = torch.zeros((Tt,Tt))
        for t in range(Tt):
            for j in range(t+1,Tt):
                tgt_mask[t,j] = float('-inf')#1
        
        device = self.vision_token.device
        #tgt_mask = tgt_mask.cuda(device)
        tgt_mask = tgt_mask.bool().cuda(device)#.view(T,T).repeat(12,1,1)
        to_decode = self.text_decoder(
            target,memory,tgt_mask).permute(1,0,2).view(L,B,-1,D)
        return self.answer_head(to_decode) # LxBxTtxV