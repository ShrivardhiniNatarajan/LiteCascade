import torch
from litecascade.models.litecascade import LiteCascade

def test_litecascade_shapes():
    model = LiteCascade(F_in=115, d=16, c=24, hidden=24, K=11, variant='R')
    x = torch.randn(4, 10, 115)
    l1, l2 = model(x, mode="train")
    assert l1.shape == (4, 11)
    assert l2.shape == (4, 11)

def test_parameter_count():
    model_r = LiteCascade(F_in=115, d=16, c=24, hidden=24, K=11, variant='R')
    params_r = sum(p.numel() for p in model_r.parameters())
    print(f"Variant R params: {params_r} (Design estimate: ~20K)")
    assert 14000 < params_r < 26000, f"Real params {params_r} differs by >30% from 20K"
    
    model_t = LiteCascade(F_in=115, d=16, c=24, K=11, variant='T')
    params_t = sum(p.numel() for p in model_t.parameters())
    print(f"Variant T params: {params_t} (Design estimate lower)")
    assert params_t < params_r

def test_pip_gradients():
    model = LiteCascade(F_in=115, d=16)
    x = torch.randn(2, 10, 115)
    l1, l2 = model(x, mode="train")
    loss = l1.sum() + l2.sum()
    loss.backward()
    assert model.pip.linear.weight.grad is not None
    
def test_gate_logic():
    model = LiteCascade(F_in=115, d=16, K=11)
    x = torch.randn(4, 10, 115)
    
    # All exit at sentinel
    preds, mask, _ = model(x, mode="infer", tau_b=0.0, tau_m=0.0)
    assert mask.all()
    
    # None exit at sentinel
    preds, mask, _ = model(x, mode="infer", tau_b=1.0, tau_m=1.0)
    assert not mask.any()
    
    # Infer outputs equal analyst outputs for non-exited
    _, l2 = model(x, mode="train")
    preds2 = torch.argmax(l2, dim=-1)
    assert torch.equal(preds, preds2)
