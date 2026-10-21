import numpy as np
import pytest
import torch
import torch.nn.functional as F

from litecascade.compress.prune import prune_litecascade
from litecascade.compress.quantize import build_int8_skeleton, convert_qat, prepare_qat, quantize_dynamic_int8
from litecascade.eval.profiler import count_macs, profile_cascade, serialized_size_kb
from litecascade.models.litecascade import LiteCascade
from litecascade.train.loss import CascadeLoss


def test_pip_pca_init_implements_eq1():
    rng = np.random.default_rng(0)
    W = rng.normal(size=(8, 20)).astype(np.float32)
    mu = rng.normal(size=20).astype(np.float32)
    m = LiteCascade(F_in=20, d=8)
    m.init_pip_from_pca(W, mu)
    x = torch.randn(3, 5, 20)
    expected = (x - torch.from_numpy(mu)) @ torch.from_numpy(W).T
    assert torch.allclose(m.pip(x), expected, atol=1e-5)


def test_temperatures_are_buffers_not_trained():
    m = LiteCascade(F_in=20, d=8)
    names = {n for n, _ in m.named_parameters()}
    assert "T1" not in names and "T2" not in names
    assert "T1" in m.state_dict() and "T2" in m.state_dict()


def test_paper_parameter_and_mac_targets_F115():
    """Design numbers from the paper (Sec. IV.C-D): Sentinel exit < 2K params, ~20K total,
    ~0.06M / ~0.18M MACs for the Sentinel-only / full path (F=115, T=10, d=32, c=24)."""
    m = LiteCascade(F_in=115, d=32, c=24, hidden=24, K=2, variant="R")
    pb = m.param_breakdown()
    assert pb["sentinel_exit_only"] < 2000
    assert 14000 < pb["total"] < 26000
    prof = profile_cascade(m, 10, 115, rho=0.8, latency_runs=5)
    assert 40_000 < prof["macs_sentinel_path"] < 80_000
    assert 140_000 < prof["macs_full"] < 220_000
    assert LiteCascade(F_in=115, d=32, K=2, variant="T").param_breakdown()["total"] < pb["total"]


def test_cascade_predict_skips_analyst_for_exited_windows():
    torch.manual_seed(0)
    m = LiteCascade(F_in=12, d=8, K=2).eval()
    seen = []
    m.analyst_blk1.register_forward_hook(lambda mod, inp, out: seen.append(inp[0].shape[0]))
    x = torch.randn(32, 10, 12)
    out = m.cascade_predict(x, tau_b=0.0, tau_m=0.0)  # everything exits
    assert out["exit_mask"].all() and out["n_analyst"] == 0 and seen == []
    out = m.cascade_predict(x, tau_b=0.6, tau_m=0.6)
    n_left = int((~out["exit_mask"]).sum())
    assert out["n_analyst"] == n_left and (seen == [n_left] if n_left else seen == [])
    # predictions of non-exited windows equal the Analyst's argmax
    l1, l2 = m(x, mode="train")
    idx = ~out["exit_mask"]
    assert torch.equal(out["preds"][idx], l2.argmax(-1)[idx])
    assert torch.equal(out["preds"][~idx], l1.argmax(-1)[~idx])


def test_verify_alerts_sends_all_alerts_to_analyst():
    p1 = torch.tensor([[0.99, 0.01], [0.01, 0.99]])
    assert LiteCascade.exit_mask(p1, 0.9, 0.9).tolist() == [True, True]
    assert LiteCascade.exit_mask(p1, 0.9, 0.9, verify_alerts=True).tolist() == [True, False]


def test_self_kd_direction_and_gradient_flow():
    torch.manual_seed(0)
    l1 = torch.randn(6, 3, requires_grad=True)
    l2 = torch.randn(6, 3, requires_grad=True)
    y = torch.randint(0, 3, (6,))
    loss_fn = CascadeLoss(gamma=0.0, delta=1.0)
    loss, parts = loss_fn(l1, l2, y, return_parts=True)
    p2 = F.softmax(l2, -1)
    kl_p2_p1 = (p2 * (p2.log() - F.log_softmax(l1, -1))).sum(-1).mean()
    assert abs(parts["kd_21"] - kl_p2_p1.item()) < 1e-5  # KL(p2 || p1)
    # Gradient of the self-KD term reaches the Sentinel only.
    only_kd = loss_fn(l1, l2, y) - F.cross_entropy(l1, y) - F.cross_entropy(l2, y)
    g1, g2 = torch.autograd.grad(only_kd, [l1, l2], allow_unused=True)
    assert g1.abs().sum() > 0 and (g2 is None or g2.abs().sum() == 0)


def test_teacher_kd_term_matches_hinton():
    torch.manual_seed(1)
    l2, t = torch.randn(5, 2), torch.randn(5, 2)
    tau = 3.0
    _, parts = CascadeLoss(gamma=1.0, delta=0.0, tau_kd=tau)(torch.randn(5, 2), l2, torch.zeros(5, dtype=torch.long), t, True)
    pt = F.softmax(t / tau, -1)
    ref = tau**2 * (pt * (pt.log() - F.log_softmax(l2 / tau, -1))).sum(-1).mean()
    assert abs(parts["kd_t2"] - ref.item()) < 1e-5


@pytest.mark.parametrize("variant", ["R", "T"])
def test_structured_pruning_shrinks_model(variant):
    m = LiteCascade(F_in=30, d=16, K=2, variant=variant).eval()
    p = prune_litecascade(m, 0.5).eval()
    assert p.param_breakdown()["total"] < m.param_breakdown()["total"]
    assert p.analyst_hidden == (24, 24) and p.sentinel_c == 12
    x = torch.randn(4, 10, 30)
    assert p(x)[1].shape == (4, 2)
    assert count_macs(p, x[:1]) < count_macs(m, x[:1])
    # config round-trip rebuilds the pruned architecture
    p2 = LiteCascade(**p.config())
    p2.load_state_dict(p.state_dict())


@pytest.mark.parametrize("variant", ["R", "T"])
def test_qat_int8_roundtrip(variant):
    torch.manual_seed(0)
    m = LiteCascade(F_in=30, d=16, K=2, variant=variant)
    q = prepare_qat(m)
    x = torch.randn(16, 10, 30)
    l1, l2 = q(x, mode="train")
    (l1.sum() + l2.sum()).backward()
    int8 = convert_qat(q)
    assert serialized_size_kb(int8) < serialized_size_kb(m)
    out = int8.cascade_predict(x, 0.7, 0.7)
    assert out["preds"].shape == (16,)
    skel = build_int8_skeleton(m.config(), "qat")
    skel.load_state_dict(int8.state_dict())
    assert torch.allclose(skel(x)[1], int8(x)[1])
    dyn = quantize_dynamic_int8(m)
    assert dyn(x)[0].shape == (16, 2)


def test_rnn_macs_are_counted():
    """fvcore counts LSTM as 0 MACs by default; the custom handle must count 4*H*(I+H) per step."""
    lstm = torch.nn.LSTM(16, 32, batch_first=True)

    class Wrap(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l = lstm

        def forward(self, x):
            return self.l(x)[0]

    assert count_macs(Wrap(), torch.randn(1, 10, 16)) == 10 * 4 * 32 * (16 + 32)
