import numpy as np
import torch

from litecascade.eval.gate import (
    NEVER,
    calibrate_arrays,
    clopper_pearson_upper,
    evaluate_gate_arrays,
    gate_exit_mask,
    max_allowed_misses,
    min_calibration_size,
    select_tau_b,
    select_tau_m,
)
from litecascade.models.litecascade import LiteCascade


def test_min_calibration_size_rule_of_three():
    # ln(0.05)/ln(0.995) = 597.6 -> 598 malicious calibration windows needed for eps=0.5%
    assert min_calibration_size(0.005, 0.05) == 598
    assert clopper_pearson_upper(0, 598, 0.05) <= 0.005 < clopper_pearson_upper(0, 597, 0.05)


def test_max_allowed_misses_matches_bound():
    n, eps = 2000, 0.005
    m = max_allowed_misses(n, eps)
    assert clopper_pearson_upper(m, n) <= eps < clopper_pearson_upper(m + 1, n)


def _toy(n_mal, n_ben, rng):
    labels = np.r_[np.ones(n_mal, int), np.zeros(n_ben, int)]
    p0 = np.r_[rng.beta(2, 5, n_mal), rng.beta(5, 2, n_ben)].astype(np.float32)
    p1 = np.stack([p0, 1 - p0], 1)
    analyst = labels.copy()  # a perfect Analyst
    return p1, analyst, labels


def test_tau_b_infeasible_for_small_calibration_set():
    rng = np.random.default_rng(0)
    p1, analyst, labels = _toy(100, 100, rng)
    out = select_tau_b(p1, analyst, labels, epsilon=0.005)
    assert out["feasible"] is False and out["tau_b"] == NEVER
    assert not gate_exit_mask(p1, out["tau_b"], NEVER).any()


def test_tau_b_respects_bound_on_calibration_and_counts_analyst_condition():
    rng = np.random.default_rng(1)
    p1, analyst, labels = _toy(3000, 3000, rng)
    out = select_tau_b(p1, analyst, labels, epsilon=0.01)
    assert out["feasible"]
    exits = gate_exit_mask(p1, out["tau_b"], NEVER)
    m = int((exits & (labels == 1) & (analyst == 1)).sum())
    assert m == out["m"] <= out["m_max"]
    assert clopper_pearson_upper(m, 3000) <= 0.01
    # A threshold just below tau_b would admit one more miss (tau_b is the smallest admissible one).
    lower = np.nextafter(np.float32(out["tau_b"]), np.float32(0))
    m_lower = int((gate_exit_mask(p1, lower, NEVER) & (labels == 1) & (analyst == 1)).sum())
    assert m_lower > out["m_max"] or out["tau_b"] <= 0.5

    # If the Analyst would also miss those windows, they are not exit-induced misses -> lower tau_b allowed.
    blind = np.zeros_like(analyst)
    assert select_tau_b(p1, blind, labels, epsilon=0.01)["tau_b"] <= out["tau_b"]
    # Conservative mode ignores the Analyst and must be at least as strict.
    assert select_tau_b(p1, blind, labels, epsilon=0.01, conservative=True)["tau_b"] >= out["tau_b"]


def test_tau_b_ties_do_not_leak_exits():
    labels = np.r_[np.ones(1000, int), np.zeros(10, int)]
    p0 = np.full(1010, 0.9, np.float32)  # every malicious window has the same benign score
    p1 = np.stack([p0, 1 - p0], 1)
    out = select_tau_b(p1, labels, labels, epsilon=0.005)
    exits = gate_exit_mask(p1, out["tau_b"], NEVER)
    assert int((exits & (labels == 1)).sum()) <= out["m_max"]


def test_risk_guarantee_monte_carlo():
    """Coverage check: over many calibration draws, P(test exit-induced miss rate > eps) <= alpha (+ MC slack)."""
    rng = np.random.default_rng(42)
    eps, alpha, n_cal, trials = 0.02, 0.1, 1500, 300

    def draw(n):
        p0 = rng.beta(2, 6, n).astype(np.float32)
        miss_analyst = rng.random(n) < 0.3  # Analyst flags 70% of malicious windows
        return np.stack([p0, 1 - p0], 1), np.where(miss_analyst, 0, 1)

    # population risk curve on a huge sample
    big_p, big_a = draw(400_000)
    violations = 0
    for _ in range(trials):
        p, a = draw(n_cal)
        tau = select_tau_b(p, a, np.ones(n_cal, int), epsilon=eps, alpha=alpha)["tau_b"]
        risk = np.mean((big_p.argmax(1) == 0) & (big_p[:, 0] >= np.float32(tau)) & (big_a != 0))
        violations += risk > eps
    assert violations / trials <= alpha + 0.05, violations / trials


def test_tau_m_target_fpr():
    rng = np.random.default_rng(3)
    p1, _, labels = _toy(2000, 2000, rng)
    tau = select_tau_m(p1, labels, target_fpr=0.02)
    ben = labels == 0
    alerts = (p1.argmax(1) != 0) & (p1.max(1) >= np.float32(tau))
    assert alerts[ben].mean() <= 0.02


def test_calibrate_and_evaluate_arrays_consistent_with_model_gate():
    rng = np.random.default_rng(4)
    l1 = rng.normal(size=(800, 2)).astype(np.float32) * 3
    l2 = rng.normal(size=(800, 2)).astype(np.float32) * 3
    y = rng.integers(0, 2, 800)
    gate = calibrate_arrays(l1, l2, y, epsilon=0.05)
    rep, preds, probs, exits = evaluate_gate_arrays(l1, l2, y, gate)
    p1 = torch.softmax(torch.from_numpy(l1) / torch.tensor([gate["T1"]], dtype=torch.float32), -1)
    torch_mask = LiteCascade.exit_mask(p1, gate["tau_b"], gate["tau_m"]).numpy()
    assert np.array_equal(torch_mask, exits)
    assert 0 <= rep["rho_raw"] <= 1 and set(rep["rho_reweighted"]) == {"1pct", "2pct", "5pct"}
