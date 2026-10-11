import torch
import torch.nn as nn
import torch.nn.functional as F

class CascadeLoss(nn.Module):
    def __init__(self, gamma=0.5, delta=0.3, tau_kd=3.0, class_weights=None):
        super().__init__()
        self.gamma = gamma
        self.delta = delta
        self.tau_kd = tau_kd
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        
    def forward(self, logits1, logits2, y, teacher_logits=None):
        loss_ce1 = self.ce(logits1, y)
        loss_ce2 = self.ce(logits2, y)
        loss = loss_ce1 + loss_ce2
        
        if self.gamma > 0 and teacher_logits is not None:
            log_p2 = F.log_softmax(logits2 / self.tau_kd, dim=-1)
            p_t = F.softmax(teacher_logits / self.tau_kd, dim=-1)
            # KLDivLoss expects input in log-space and target in prob-space when log_target=False (default)
            loss_kd = F.kl_div(log_p2, p_t, reduction='batchmean')
            loss = loss + self.gamma * (self.tau_kd ** 2) * loss_kd
            
        if self.delta > 0:
            log_p2 = F.log_softmax(logits2, dim=-1)
            p1 = F.softmax(logits1.detach(), dim=-1) # don't backprop to sentinel for self-KD?
            # actually usually p2 tries to mimic p1 or p1 tries to mimic p2? "KL(p2 || p1)" means target is p1.
            loss_self_kd = F.kl_div(log_p2, p1, reduction='batchmean')
            loss = loss + self.delta * loss_self_kd
            
        return loss
