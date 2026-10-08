"""
Gate calibration and risk-controlled thresholds for LiteCascade.
"""
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import beta as beta_dist
from scipy.optimize import minimize_scalar

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clopper_pearson_upper(m, n, alpha=0.05):
    """One-sided Clopper-Pearson upper bound."""
    if n == 0:
        return 1.0
    if m == 0:
        return 1.0 - alpha ** (1.0 / n)
    return beta_dist.ppf(1 - alpha, m + 1, n - m)


class TemperatureScaler(nn.Module):
    """Temperature scaling for calibration."""
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, logits):
        return logits / self.temperature


def fit_temperature(logits, labels, max_iter=50):
    """Fit temperature on calibration set by minimising NLL via LBFGS."""
    scaler = TemperatureScaler()
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)
    
    nll_before = F.cross_entropy(logits_t, labels_t).item()
    
    optimizer = torch.optim.LBFGS([scaler.temperature], lr=0.01, max_iter=max_iter)
    
    def closure():
        optimizer.zero_grad()
        scaled = scaler(logits_t)
        loss = F.cross_entropy(scaled, labels_t)
        loss.backward()
        return loss
    
    optimizer.step(closure)
    
    nll_after = F.cross_entropy(scaler(logits_t), labels_t).item()
    
    logger.info(f"Temperature scaling: T={scaler.temperature.item():.4f}, NLL before={nll_before:.4f}, after={nll_after:.4f}")
    
    return scaler.temperature.item()


def compute_ece(probs, labels, n_bins=15):
    """Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (probs.max(axis=1) > lo) & (probs.max(axis=1) <= hi)
        if mask.sum() == 0:
            continue
        avg_confidence = probs.max(axis=1)[mask].mean()
        preds = probs.argmax(axis=1)[mask]
        avg_accuracy = (preds == labels[mask]).mean()
        ece += mask.sum() / len(labels) * abs(avg_confidence - avg_accuracy)
    return ece


def find_tau_b(model, calib_loader, epsilon=0.005, alpha=0.05, temperature=1.0):
    """
    Find tau_b: smallest tau such that CP upper bound on exit-induced miss rate <= epsilon.
    """
    model.eval()
    
    # Collect sentinel predictions and analyst predictions on calibration set
    sentinel_logits_list = []
    analyst_logits_list = []
    labels_list = []
    
    with torch.no_grad():
        for X, y in calib_loader:
            l1, l2 = model(X, mode="train")
            sentinel_logits_list.append(l1.cpu())
            analyst_logits_list.append(l2.cpu())
            labels_list.append(y.cpu())
    
    sentinel_logits = torch.cat(sentinel_logits_list, dim=0).numpy()
    analyst_logits = torch.cat(analyst_logits_list, dim=0).numpy()
    labels = torch.cat(labels_list, dim=0).numpy()
    
    # Apply temperature scaling to sentinel
    sentinel_probs = torch.softmax(torch.tensor(sentinel_logits) / temperature, dim=-1).numpy()
    analyst_preds = np.argmax(analyst_logits, axis=1)
    
    # Malicious calibration windows
    malicious_mask = labels > 0
    N_m = malicious_mask.sum()
    
    if N_m == 0:
        logger.warning("No malicious samples in calibration set")
        return 0.5, 0, 0
    
    # Search over fine grid of tau
    tau_grid = np.linspace(0.0, 1.0, 1001)
    best_tau = 1.0
    
    for tau in tau_grid:
        # Count exit-induced misses: malicious windows that exit benign at sentinel
        sentinel_preds = np.argmax(sentinel_probs, axis=1)
        # Would exit: pred==0 and p[0] >= tau
        would_exit = (sentinel_preds == 0) & (sentinel_probs[:, 0] >= tau)
        
        # Of those malicious that would exit benign, how many does analyst classify as malicious?
        exit_miss = would_exit & malicious_mask
        m = exit_miss.sum()
        
        U_CP = clopper_pearson_upper(int(m), int(N_m), alpha)
        
        if U_CP <= epsilon:
            best_tau = tau
            break
    
    return best_tau, int(m) if best_tau < 1.0 else 0, int(N_m)


def find_tau_m(model, calib_loader, target_fpr=0.01, temperature=1.0):
    """Find tau_m for target false-positive rate on calibration."""
    model.eval()
    
    sentinel_logits_list = []
    labels_list = []
    
    with torch.no_grad():
        for X, y in calib_loader:
            l1, l2 = model(X, mode="train")
            sentinel_logits_list.append(l1.cpu())
            labels_list.append(y.cpu())
    
    sentinel_logits = torch.cat(sentinel_logits_list, dim=0).numpy()
    labels = torch.cat(labels_list, dim=0).numpy()
    
    sentinel_probs = torch.softmax(torch.tensor(sentinel_logits) / temperature, dim=-1).numpy()
    
    benign_mask = labels == 0
    N_b = benign_mask.sum()
    
    if N_b == 0:
        return 0.5
    
    tau_grid = np.linspace(0.0, 1.0, 1001)
    best_tau = 0.5
    
    for tau in tau_grid:
        sentinel_preds = np.argmax(sentinel_probs, axis=1)
        # False positives: benign samples predicted as malicious with high confidence
        would_alert = (sentinel_preds != 0) & (sentinel_probs.max(axis=1) >= tau)
        fp_rate = would_alert[benign_mask].sum() / N_b
        
        if fp_rate <= target_fpr:
            best_tau = tau
            break
    
    return best_tau


def evaluate_cascade(model, test_loader, tau_b, tau_m, temperature=1.0):
    """Evaluate full cascade on test set."""
    model.eval()
    
    all_preds = []
    all_labels = []
    exit_flags = []
    analyst_preds_all = []
    
    with torch.no_grad():
        for X, y in test_loader:
            l1, l2 = model(X, mode="train")
            
            probs1 = torch.softmax(l1 / temperature, dim=-1)
            preds1 = torch.argmax(probs1, dim=-1)
            preds2 = torch.argmax(l2, dim=-1)
            
            for i in range(len(X)):
                p1 = probs1[i]
                pred = preds1[i].item()
                
                exited = False
                if pred == 0 and p1[0].item() >= tau_b:
                    exited = True
                elif pred != 0 and p1.max().item() >= tau_m:
                    exited = True
                
                if exited:
                    all_preds.append(pred)
                else:
                    all_preds.append(preds2[i].item())
                    
                exit_flags.append(exited)
                all_labels.append(y[i].item())
                analyst_preds_all.append(preds2[i].item())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    exit_flags = np.array(exit_flags)
    analyst_preds_all = np.array(analyst_preds_all)
    
    # Exit rate
    rho_raw = exit_flags.mean()
    
    # Re-weighted exit rates for different malicious prevalence
    rho_reweighted = {}
    for prev in [0.01, 0.02, 0.05]:
        malicious_mask = all_labels > 0
        benign_mask = all_labels == 0
        
        n_mal = malicious_mask.sum()
        n_ben = benign_mask.sum()
        
        if n_mal == 0 or n_ben == 0:
            rho_reweighted[f"{int(prev*100)}pct"] = rho_raw
            continue
            
        # Importance weights
        w_mal = prev / (n_mal / len(all_labels))
        w_ben = (1 - prev) / (n_ben / len(all_labels))
        
        weights = np.where(malicious_mask, w_mal, w_ben)
        rho_reweighted[f"{int(prev*100)}pct"] = (exit_flags * weights).sum() / weights.sum()
    
    # Exit-induced recall loss
    malicious_mask = all_labels > 0
    if malicious_mask.sum() > 0:
        # Malicious samples that exited as benign
        exit_induced_misses = exit_flags & malicious_mask & (all_preds == 0)
        recall_loss = exit_induced_misses.sum() / malicious_mask.sum()
    else:
        recall_loss = 0.0
    
    # F1 of cascade vs analyst-only
    from litecascade.eval.metrics import compute_detection_metrics
    cascade_metrics = compute_detection_metrics(all_labels, all_preds)
    analyst_metrics = compute_detection_metrics(all_labels, analyst_preds_all)
    
    report = {
        "rho_raw": float(rho_raw),
        "rho_reweighted": {k: float(v) for k, v in rho_reweighted.items()},
        "recall_loss": float(recall_loss),
        "cascade_f1": float(cascade_metrics["f1_macro"]),
        "analyst_only_f1": float(analyst_metrics["f1_macro"]),
        "tau_b": float(tau_b),
        "tau_m": float(tau_m),
        "temperature": float(temperature),
    }
    
    return report


def calibrate_and_report(model, calib_loader, test_loader, run_id, epsilon=0.005):
    """Full calibration pipeline."""
    
    # 1. Collect calibration logits
    calib_logits = []
    calib_labels = []
    model.eval()
    with torch.no_grad():
        for X, y in calib_loader:
            l1, _ = model(X, mode="train")
            calib_logits.append(l1.cpu())
            calib_labels.append(y.cpu())
    
    calib_logits_np = torch.cat(calib_logits, dim=0).numpy()
    calib_labels_np = torch.cat(calib_labels, dim=0).numpy()
    
    # ECE before
    probs_before = torch.softmax(torch.tensor(calib_logits_np), dim=-1).numpy()
    ece_before = compute_ece(probs_before, calib_labels_np)
    
    # 2. Temperature scaling
    temperature = fit_temperature(calib_logits_np, calib_labels_np)
    
    probs_after = torch.softmax(torch.tensor(calib_logits_np) / temperature, dim=-1).numpy()
    ece_after = compute_ece(probs_after, calib_labels_np)
    
    logger.info(f"ECE before: {ece_before:.4f}, after: {ece_after:.4f}")
    
    # Verify argmax unchanged
    assert np.array_equal(probs_before.argmax(axis=1), probs_after.argmax(axis=1)), \
        "Temperature scaling changed argmax!"
    
    # 3. Find tau_b
    tau_b, m, N_m = find_tau_b(model, calib_loader, epsilon=epsilon, temperature=temperature)
    logger.info(f"tau_b={tau_b:.4f}, m={m}, N_m={N_m}")
    
    # 4. Find tau_m
    tau_m = find_tau_m(model, calib_loader, target_fpr=0.01, temperature=temperature)
    logger.info(f"tau_m={tau_m:.4f}")
    
    # 5. Evaluate on test
    report = evaluate_cascade(model, test_loader, tau_b, tau_m, temperature)
    report["ece_before"] = float(ece_before)
    report["ece_after"] = float(ece_after)
    report["epsilon"] = epsilon
    
    # 6. Save
    out_dir = Path(f"results/{run_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "gate_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    # Save gate thresholds
    Path("checkpoints").mkdir(exist_ok=True)
    with open(f"checkpoints/gate_{run_id}.json", "w") as f:
        json.dump({"tau_b": tau_b, "tau_m": tau_m, "temperature": temperature}, f, indent=2)
    
    logger.info(f"Gate report saved to {out_dir / 'gate_report.json'}")
    return report
