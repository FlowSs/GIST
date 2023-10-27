# -*-coding:utf-8-*-
import torch.nn as nn
import transformers

__all__ = ["roberta"]

class BERTMode(nn.Module):
    def __init__(self):
        super(BERTMode, self).__init__()

        self.BERT = transformers.RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2) 
        self.BERT.config.output_hidden_states=True

    def forward(self, input_ids, attention_mask, labels=None, hook=False):
        x = self.BERT(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        
        if hook:
            return {'pred': x['logits'].detach().cpu().numpy(), 
            'x_flat': x['hidden_states'][-1][:,0,:].detach().cpu().numpy()}
        else:
            return x['logits']

def roberta():
    
    return BERTMode()