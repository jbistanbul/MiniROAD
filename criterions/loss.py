import torch
import torch.nn as nn
import torch.nn.functional as F
from criterions.loss_builder import CRITERIONS

@CRITERIONS.register('NONUNIFORM')
class OadLoss(nn.Module):
    
    def __init__(self, cfg, reduction='mean'):
        super(OadLoss, self).__init__()
        self.reduction = reduction
        self.num_classes = cfg['num_classes']
        self.loss = self.end_loss

    def end_loss(self, out_dict, target):
        # logits: (B, seq, K) target: (B, seq, K)
        logits = out_dict['logits']
        logits = logits[:,-1,:].contiguous()
        target = target[:,-1,:].contiguous()
        ce_loss = self.mlce_loss(logits, target)
        return ce_loss

    def mlce_loss(self, logits, target):
        '''
        multi label cross entropy loss. 
        logits: (B, K) target: (B, K) 
        '''
        logsoftmax = nn.LogSoftmax(dim=-1).to(logits.device)
        output = torch.sum(-F.normalize(target) * logsoftmax(logits), dim=1) # B
        if self.reduction == 'mean':
            loss = torch.mean(output)
        elif self.reduction == 'sum':
            loss = torch.sum(output)
        return loss

    def forward(self, out_dict, target): 
        return self.loss(out_dict, target)
    

@CRITERIONS.register('ANTICIPATION')
class OadAntLoss(nn.Module):
    
    def __init__(self, cfg, reduction='sum'):
        super(OadAntLoss, self).__init__()
        self.reduction = reduction
        self.loss = self.anticipation_loss
        self.num_classes = cfg['num_classes']

    def anticipation_loss(self, out_dict, target, ant_target):
        anticipation_logits = out_dict['anticipation_logits']
        pred_anticipation_logits = anticipation_logits[:,-1,:,:].contiguous().view(-1, self.num_classes)
        anticipation_logit_targets = ant_target.view(-1, self.num_classes)
        ant_loss = self.mlce_loss(pred_anticipation_logits, anticipation_logit_targets)
        return ant_loss 

    def ce_loss(self, out_dict, target):
        # logits: (B, seq, K) target: (B, seq, K)
        logits = out_dict['logits']
        logits = logits[:,-1,:].contiguous()
        target = target[:,-1,:].contiguous()
        ce_loss = self.mlce_loss(logits, target)
        return ce_loss

    def mlce_loss(self, logits, target):
        '''
        multi label cross entropy loss. 
        logits: (B, K) target: (B, K) 
        '''
        logsoftmax = nn.LogSoftmax(dim=-1).to(logits.device)
        output = torch.sum(-F.normalize(target) * logsoftmax(logits), dim=1) # B
        if self.reduction == 'mean':
            loss = torch.mean(output)
        elif self.reduction == 'sum':
            loss = torch.sum(output)

        return loss

    def forward(self, out_dict, target, ant_target): 
        return self.loss(out_dict, target, ant_target)
