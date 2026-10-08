import torch
import torch.nn.functional as F
from litecascade.train.loss import CascadeLoss

def test_kd_loss_zero():
    loss_fn = CascadeLoss(gamma=0.5, delta=0.3, tau_kd=3.0)
    
    # student == teacher logits
    logits1 = torch.randn(4, 2)
    logits2 = logits1.clone()
    teacher_logits = logits2.clone()
    y = torch.randint(0, 2, (4,))
    
    # If student == teacher, KL is 0. 
    # Also if logits1 == logits2, self-KD is 0.
    loss = loss_fn(logits1, logits2, y, teacher_logits)
    ce1 = F.cross_entropy(logits1, y)
    ce2 = F.cross_entropy(logits2, y)
    
    # Assert loss matches ce1 + ce2 exactly
    assert torch.isclose(loss, ce1 + ce2, atol=1e-5)
    
def test_kd_loss_reduction():
    loss_fn = CascadeLoss(gamma=0.0, delta=0.0)
    logits1 = torch.randn(4, 2)
    logits2 = torch.randn(4, 2)
    teacher_logits = torch.randn(4, 2)
    y = torch.randint(0, 2, (4,))
    
    loss = loss_fn(logits1, logits2, y, teacher_logits)
    ce1 = F.cross_entropy(logits1, y)
    ce2 = F.cross_entropy(logits2, y)
    assert torch.isclose(loss, ce1 + ce2, atol=1e-5)
