import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class Bert(nn.Module):
    def __init__(self,cfg=None):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') 
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def forward(self,sentences,device=None):
        token_inputs = self.tokenizer(
            sentences,
            padding=True,
            return_tensors='pt')
        if device is None:
            token_inputs = {k:v.cuda() for k,v in token_inputs.items()}
        else:
            token_inputs = {k:v.cuda(device) for k,v in token_inputs.items()}

        outputs = self.model(**token_inputs)
        return outputs[0], token_inputs # BxTxD (BxTx768)


if __name__=='__main__':
    bert = Bert(None)
    bert(['How do you do?','I am fine thank you.'])