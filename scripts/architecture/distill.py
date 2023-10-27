# -*-coding:utf-8-*-
import torch.nn as nn
import torch
import transformers

__all__ = ["distill"]

class DistillBERT(nn.Module):
    def __init__(self):
        super(DistillBERT, self).__init__()

        self.model = transformers.DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
        self.model.config.output_hidden_states = True

    def forward(self, input_ids, attention_mask, labels=None, hook=False):
        x = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        if hook:
            return {'pred': x['logits'].detach().cpu().numpy(), 
            'x_flat': x['hidden_states'][-1][:,0,:].detach().cpu().numpy()}
        else:
            return x['logits']

def distill():
    
    return DistillBERT()