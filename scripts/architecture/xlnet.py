# -*-coding:utf-8-*-
import torch.nn as nn
import transformers

__all__ = ["xlnet"]

class XLNET(nn.Module):
    def __init__(self):
        super(XLNET, self).__init__()

        self.model = transformers.XLNetForSequenceClassification.from_pretrained("xlnet-base-cased", num_labels=2)
        self.model.config.output_hidden_states=True

    def forward(self, input_ids, attention_mask, labels=None, hook=False):
        x = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)      
        
        if hook:
            # x_flat is -1 instead of 0 for XLNet since [CLS] token is last
            return {'pred': x['logits'].detach().cpu().numpy(), 
            'x_flat': x['hidden_states'][-1][:,-1,:].detach().cpu().numpy()}
        else:
            return x['logits']

def xlnet():
    
    return XLNET()