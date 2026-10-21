"""Gate calibration and risk-controlled thresholds for LiteCascade (paper Sec. IV.E, eqs. 2-4).

Temperature scaling (Guo et al., 2017): T1 (Sentinel) and T2 (Analyst) minimise NLL on the
calibration split; T > 0 is enforced by optimising log T, so argmax is unchanged.

tau_b (eq. 4): an *exit-induced miss* is a malicious calibration window that exits benign at the Sentinel
while the Analyst would have flagged it (Analyst argmax != 0). With m(tau) such windows among N_m
malicious calibration windows

    tau_b = min{ tau : U_CP(m(tau), N_m; alpha) <= epsilon }

m(tau) is non-increasing in tau, so the admissible set is an interval [tau_b, inf). Choosing its
smallest element equals fixed-sequence testing of H_tau: risk(tau) > epsilon from tau = inf downwards,
stopping at the first non-rejection, which controls the family-wise error at alpha without a multiplicity
correction. Hence, under exchangeability of calibration and test windows, with probability >= 1 - alpha
over the calibration draw the exit-induced miss rate on test data is <= epsilon.

Implementation detail: probabilities are computed in float32 exactly as at inference time and thresholds
are placed at float32 ``nextafter`` of a calibration score, so ties cannot leak extra exits. If even
m = 0 violates the bound (N_m < ln(alpha)/ln(1-epsilon), ~598 for eps=0.005, alpha=0.05) the
benign exit is disabled (tau_b = NEVER) and the result is flagged infeasible.

tau_m: smallest threshold whose empirical Sentinel false-alarm exit rate on benign calibration windows
is <= target_fpr. This is an empirical choice WITHOUT a statistical guarantee; ``verify_alerts=True``
(tau_m = NEVER) sends every alert to the Analyst instead.
"""

import json
import logging
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import beta as beta_dist

from litecascade.eval.metrics import compute_detection_metrics

logger = logging.getLogger(__name__)

NEVER = 2.0  # probabilities are <= 1, so a threshold of 2 disables that exit (JSON-safe, unlike inf)


# --------------------------------------------------------------------------------------------- bounds
def clopper_pearson_upper(m, n, alpha=0.05):
    """One-sided (1 - alpha) Clopper-Pearson upper bound for a binomial proportion with m successes in n."""
    if n == 0:
        return 1.0
    if m >= n:
        return 1.0
    if m == 0:
        return 1.0 - alpha ** (1.0 / n)
    return float(beta_dist.ppf(1 - alpha, m + 1, n - m))


def min_calibration_size(epsilon: float, alpha: float = 0.05) -> int:
    """Smallest N_m for which m = 0 satisfies U_CP <= epsilon."""
    return int(math.ceil(math.log(alpha) / math.log(1 - epsilon)))


def max_allowed_misses(n: int, epsilon: float, alpha: float = 0.05) -> int:
    """Largest m with U_CP(m, n) <= epsilon, or -1 if none."""
    m = -1
    while m + 1 <= n and clopper_pearson_upper(m + 1, n, alpha) <= epsilon:
        m += 1
    return m


# --------------------------------------------------------------------------------------------- temperature
class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_t = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self):
        return self.log_t.exp()

    def forward(self, logits):
        return logits / self.temperature


def fit_temperature(logits, labels, max_iter=100):
    """Fit T > 0 on calibration logits by minimising NLL with L-BFGS (over log T)."""
    logits_t = torch.as_tensor(np.asarray(logits), dtype=torch.float32)
    labels_t = torch.as_tensor(np.asarray(labels), dtype=torch.long)
    if len(labels_t) == 0:
        return 1.0
    scaler = TemperatureScaler()
    optimizer = torch.optim.LBFGS([scaler.log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(scaler(logits_t), labels_t)
        loss.backward()
        return loss

    nll_before = F.cross_entropy(logits_t, labels_t).item()
    optimizer.step(closure)
    T = float(scaler.temperature.item())
    if not np.isfinite(T) or T <= 0:
        T = 1.0
    nll_after = F.cross_entropy(logits_t / T, labels_t).item()
    if nll_after > nll_before:  # never make calibration worse than T = 1
        T = 1.0
    logger.info(f"Temperature scaling: T={T:.4f}, NLL before={nll_before:.4f}, after={min(nll_after, nll_before):.4f}")
    return T


def compute_ece(probs, labels, n_bins=15):
    """Expected Calibration Error (top-label)."""
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    if len(labels) == 0:
        return 0.0
    conf = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(conf[mask].mean() - (preds[mask] == labels[mask]).mean())
    return float(ece)


def softmax_t(logits, T: float) -> np.ndarray:
    """float32 softmax(logits / T), bit-compatible with the model's torch inference path."""
    return torch.softmax(
        torch.as_tensor(np.asarray(logits), dtype=torch.float32) / torch.tensor([T], dtype=torch.float32), -1
    ).numpy()


# --------------------------------------------------------------------------------------------- thresholds
def _just_above(v) -> float:
    return float(np.nextafter(np.float32(v), np.float32(np.inf)))


def select_tau_b(p1, analyst_pred, labels, epsilon=0.005, alpha=0.05, conservative=False) -> dict:
    """Risk-controlled benign threshold (eq. 4). ``conservative=True`` counts every malicious window that
    exits benign as a miss (ignores whether the Analyst would have caught it)."""
    p1 = np.asarray(p1, dtype=np.float32)
    labels = np.asarray(labels)
    malicious = labels > 0
    n_m = int(malicious.sum())
    pred1 = p1.argmax(axis=1)
    eligible = malicious & (pred1 == 0)
    if not conservative:
        eligible &= np.asarray(analyst_pred) != 0
    scores = np.sort(p1[eligible, 0])[::-1]  # descending benign confidence of potential misses

    m_max = max_allowed_misses(n_m, epsilon, alpha) if n_m > 0 else -1
    out = {"N_m": n_m, "m_max": int(m_max), "epsilon": epsilon, "alpha": alpha, "min_N_m": min_calibration_size(epsilon, alpha)}
    if m_max < 0:
        out.update(tau_b=NEVER, feasible=False, m=0, ucp=clopper_pearson_upper(0, n_m, alpha))
        return out
    if len(scores) <= m_max:
        tau = 0.0  # even exiting every benign-predicted window keeps m <= m_max
    else:
        tau = _just_above(scores[m_max])
    m = int((eligible & (p1[:, 0] >= np.float32(tau)) & (pred1 == 0)).sum())
    out.update(tau_b=float(tau), feasible=True, m=m, ucp=clopper_pearson_upper(m, n_m, alpha))
    return out


def select_tau_m(p1, labels, target_fpr=0.01) -> float:
    """Smallest tau whose empirical rate of benign windows exiting as malicious is <= target_fpr."""
    p1 = np.asarray(p1, dtype=np.float32)
    labels = np.asarray(labels)
    benign = labels == 0
    n_b = int(benign.sum())
    if n_b == 0:
        return NEVER
    pred1 = p1.argmax(axis=1)
    scores = np.sort(p1[benign & (pred1 != 0)].max(axis=1))[::-1]
    k = int(math.floor(target_fpr * n_b))
    if len(scores) <= k:
        return 0.0
    return _just_above(scores[k])


def gate_exit_mask(p1, tau_b, tau_m, verify_alerts=False) -> np.ndarray:
    p1 = np.asarray(p1, dtype=np.float32)
    pred1 = p1.argmax(axis=1)
    mask = (pred1 == 0) & (p1[:, 0] >= np.float32(tau_b))
    if not verify_alerts:
        mask |= (pred1 != 0) & (p1.max(axis=1) >= np.float32(tau_m))
    return mask


# --------------------------------------------------------------------------------------------- evaluation
@torch.no_grad()
def collect_outputs(model, loader) -> dict:
    """Raw logits of both exits + labels over a loader (offline analysis; runs both heads)."""
    model.eval()
    l1s, l2s, ys = [], [], []
    for batch in loader:
        X, y = batch[0], batch[1]
        l1, l2 = model(X, mode="train")
        l1s.append(l1.float().cpu())
        l2s.append(l2.float().cpu())
        ys.append(y.cpu())
    if not ys:
        return {"l1": np.zeros((0, 2), np.float32), "l2": np.zeros((0, 2), np.float32), "y": np.zeros(0, np.int64)}
    return {"l1": torch.cat(l1s).numpy(), "l2": torch.cat(l2s).numpy(), "y": torch.cat(ys).numpy()}


def reweighted_exit_rates(exit_flags, labels, prevalences=(0.01, 0.02, 0.05)) -> dict:
    exit_flags = np.asarray(exit_flags, dtype=float)
    malicious = np.asarray(labels) > 0
    out = {}
    for prev in prevalences:
        key = f"{prev * 100:g}pct"
        if malicious.all() or (~malicious).all():
            out[key] = float(exit_flags.mean()) if len(exit_flags) else 0.0
            continue
        out[key] = float(prev * exit_flags[malicious].mean() + (1 - prev) * exit_flags[~malicious].mean())
    return out


def evaluate_gate_arrays(l1, l2, labels, gate: dict, verify_alerts=False) -> dict:
    """Offline cascade evaluation from both heads' logits (used for reports and ablations)."""
    labels = np.asarray(labels)
    p1 = softmax_t(l1, gate.get("T1", 1.0))
    p2 = softmax_t(l2, gate.get("T2", 1.0))
    exit_flags = gate_exit_mask(p1, gate["tau_b"], gate["tau_m"], verify_alerts)
    final_probs = np.where(exit_flags[:, None], p1, p2)
    preds = final_probs.argmax(axis=1)
    pred1, pred2 = p1.argmax(axis=1), p2.argmax(axis=1)
    malicious = labels > 0
    n_m = int(malicious.sum())
    exit_benign = exit_flags & (pred1 == 0)
    report = {
        "n": int(len(labels)),
        "rho_raw": float(exit_flags.mean()) if len(labels) else 0.0,
        "rho_reweighted": reweighted_exit_rates(exit_flags, labels),
        "exit_induced_miss_rate": float((exit_benign & malicious & (pred2 != 0)).sum() / n_m) if n_m else 0.0,
        "malicious_exited_benign_rate": float((exit_benign & malicious).sum() / n_m) if n_m else 0.0,
        "cascade": compute_detection_metrics(labels, preds),
        "sentinel_only": compute_detection_metrics(labels, pred1),
        "analyst_only": compute_detection_metrics(labels, pred2),
        "tau_b": float(gate["tau_b"]),
        "tau_m": float(gate["tau_m"]),
        "T1": float(gate.get("T1", 1.0)),
        "T2": float(gate.get("T2", 1.0)),
    }
    report["recall_loss_vs_analyst"] = report["analyst_only"]["recall_malicious"] - report["cascade"]["recall_malicious"]
    report["cascade_f1"] = report["cascade"]["f1_macro"]
    report["analyst_only_f1"] = report["analyst_only"]["f1_macro"]
    return report, preds, final_probs, exit_flags


def calibrate_arrays(
    l1, l2, labels, epsilon=0.005, alpha=0.05, target_fpr=0.01, verify_alerts=False, temperature=True, conservative=False
) -> dict:
    """Fit T1, T2, tau_b, tau_m on calibration outputs. Returns a JSON-serialisable gate config."""
    labels = np.asarray(labels)
    T1 = fit_temperature(l1, labels) if temperature else 1.0
    T2 = fit_temperature(l2, labels) if temperature else 1.0
    p1 = softmax_t(l1, T1)
    analyst_pred = np.asarray(l2).argmax(axis=1)
    tb = select_tau_b(p1, analyst_pred, labels, epsilon, alpha, conservative)
    tau_m = NEVER if verify_alerts else select_tau_m(p1, labels, target_fpr)
    gate = {
        "T1": T1,
        "T2": T2,
        "tau_b": tb["tau_b"],
        "tau_m": tau_m,
        "verify_alerts": bool(verify_alerts),
        "target_fpr": target_fpr,
        "feasible": tb["feasible"],
        "N_m": tb["N_m"],
        "m_max": tb["m_max"],
        "m_calib": tb["m"],
        "ucp_calib": tb["ucp"],
        "min_N_m": tb["min_N_m"],
        "epsilon": epsilon,
        "alpha": alpha,
        "conservative": conservative,
        "temperature_scaling": bool(temperature),
        "ece_sentinel_before": compute_ece(softmax_t(l1, 1.0), labels),
        "ece_sentinel_after": compute_ece(p1, labels),
    }
    if not tb["feasible"]:
        logger.warning(
            f"tau_b infeasible: N_m={tb['N_m']} malicious calibration windows < {tb['min_N_m']} needed for "
            f"epsilon={epsilon}, alpha={alpha}. Benign early exit disabled."
        )
    return gate


def apply_gate_to_model(model, gate: dict):
    model.T1.fill_(gate["T1"])
    model.T2.fill_(gate["T2"])
    model.verify_alerts = bool(gate.get("verify_alerts", False))


# --------------------------------------------------------------------------------------------- legacy API
def find_tau_b(model, calib_loader, epsilon=0.005, alpha=0.05, temperature=1.0):
    out = collect_outputs(model, calib_loader)
    tb = select_tau_b(softmax_t(out["l1"], temperature), out["l2"].argmax(axis=1), out["y"], epsilon, alpha)
    return tb["tau_b"], tb["m"], tb["N_m"]


def find_tau_m(model, calib_loader, target_fpr=0.01, temperature=1.0):
    out = collect_outputs(model, calib_loader)
    return select_tau_m(softmax_t(out["l1"], temperature), out["y"], target_fpr)


def evaluate_cascade(model, test_loader, tau_b, tau_m, temperature=1.0):
    out = collect_outputs(model, test_loader)
    report, *_ = evaluate_gate_arrays(out["l1"], out["l2"], out["y"], {"tau_b": tau_b, "tau_m": tau_m, "T1": temperature})
    return report


def calibrate_and_report(
    model, calib_loader, test_loader, run_id, epsilon=0.005, alpha=0.05, target_fpr=0.01, out_root="results"
):
    cal = collect_outputs(model, calib_loader)
    gate = calibrate_arrays(cal["l1"], cal["l2"], cal["y"], epsilon=epsilon, alpha=alpha, target_fpr=target_fpr)
    apply_gate_to_model(model, gate)
    test = collect_outputs(model, test_loader)
    report, *_ = evaluate_gate_arrays(test["l1"], test["l2"], test["y"], gate)
    report["gate"] = gate
    out_dir = Path(out_root) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "gate_report.json", "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Gate report saved to {out_dir / 'gate_report.json'}")
    return report
