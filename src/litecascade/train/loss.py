"""Joint cascade objective, paper eq. 5:

    L = CE(p1, y) + CE(p2, y) + gamma * KD_{T->2} + delta * KD_{2->1}

    KD_{T->2} = tau^2 * KL( softmax(t / tau) || softmax(s2 / tau) )   teacher -> Analyst
    KD_{2->1} = KL( p2 || p1 )                                         Analyst -> Sentinel (self-distillation)

F.kl_div(log_q, p) computes KL(p || q). For KD_{2->1} the Analyst distribution p2 is the (detached)
target and gradients flow into the Sentinel only, so self-distillation cannot drag the Analyst
towards the weaker Sentinel.
"""

import torch.nn as nn
import torch.nn.functional as F


class CascadeLoss(nn.Module):
    def __init__(self, gamma=0.5, delta=0.3, tau_kd=3.0, class_weights=None):
        super().__init__()
        self.gamma = gamma
        self.delta = delta
        self.tau_kd = tau_kd
        self.ce = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, logits1, logits2, y, teacher_logits=None, return_parts: bool = False):
        parts = {"ce1": self.ce(logits1, y), "ce2": self.ce(logits2, y)}
        loss = parts["ce1"] + parts["ce2"]

        if self.gamma > 0 and teacher_logits is not None:
            log_p2 = F.log_softmax(logits2 / self.tau_kd, dim=-1)
            p_t = F.softmax(teacher_logits / self.tau_kd, dim=-1)
            parts["kd_t2"] = (self.tau_kd**2) * F.kl_div(log_p2, p_t, reduction="batchmean")
            loss = loss + self.gamma * parts["kd_t2"]

        if self.delta > 0:
            log_p1 = F.log_softmax(logits1, dim=-1)
            p2 = F.softmax(logits2.detach(), dim=-1)
            parts["kd_21"] = F.kl_div(log_p1, p2, reduction="batchmean")
            loss = loss + self.delta * parts["kd_21"]

        if return_parts:
            return loss, {k: float(v.detach()) for k, v in parts.items()}
        return loss
