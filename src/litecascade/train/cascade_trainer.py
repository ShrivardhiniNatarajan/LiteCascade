"""Reusable training / evaluation steps for the LiteCascade chain (paper Sec. IV.F-G).

Teacher logits are computed for exactly the training windows and stored alongside them in the dataset,
so distillation targets stay aligned under shuffling.
"""

import copy
import logging
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from litecascade.data.dataset import CascadeDataset, compute_class_weights
from litecascade.eval.gate import calibrate_arrays, collect_outputs, evaluate_gate_arrays
from litecascade.eval.metrics import compute_detection_metrics
from litecascade.models.litecascade import LiteCascade
from litecascade.models.teacher import TeacherCNNBiLSTM
from litecascade.train.loss import CascadeLoss

logger = logging.getLogger(__name__)


def make_loader(X, y, batch_size=128, shuffle=False, teacher_logits=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    return DataLoader(
        CascadeDataset(X, y, teacher_logits), batch_size=batch_size, shuffle=shuffle, generator=g if shuffle else None
    )


@torch.no_grad()
def predict_logits(model, X, batch_size=512) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        out.append(model(torch.as_tensor(X[i : i + batch_size], dtype=torch.float32)).float())
    return torch.cat(out).numpy() if out else np.zeros((0, 2), np.float32)


# --------------------------------------------------------------------------------------------- teacher
def train_teacher(data, epochs=10, lr=1e-3, batch_size=128, seed=42):
    """Train the CNN-BiLSTM teacher on the train split, select on val macro-F1."""
    torch.manual_seed(seed)
    model = TeacherCNNBiLSTM(data.n_features, num_classes=data.num_classes)
    weights = compute_class_weights(data.y["train"], data.num_classes)
    crit = torch.nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    loader = make_loader(data.X["train"], data.y["train"], batch_size, shuffle=True, seed=seed)
    best, best_state = -1.0, None
    for ep in range(epochs):
        model.train()
        for X, y in loader:
            opt.zero_grad()
            loss = crit(model(X), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
        f1 = compute_detection_metrics(data.y["val"], predict_logits(model, data.X["val"]).argmax(1))["f1_macro"]
        logger.info(f"[teacher] epoch {ep + 1}/{epochs} val macro-F1={f1:.4f}")
        if f1 > best:
            best, best_state = f1, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, best


# --------------------------------------------------------------------------------------------- cascade
def build_cascade(data, variant="R", pip_init=True, d=32, c=24, hidden=24) -> LiteCascade:
    d = min(d, data.n_features)
    model = LiteCascade(F_in=data.n_features, d=d, c=c, hidden=hidden, T_len=data.T, K=data.num_classes, variant=variant)
    if pip_init:
        W, mu = data.preprocessor.pca_init_params()
        model.init_pip_from_pca(W[:d], mu)
    return model


def _val_score(model, X, y):
    out = collect_outputs(model, make_loader(X, y, 512))
    s = compute_detection_metrics(out["y"], out["l1"].argmax(1))["f1_macro"]
    a = compute_detection_metrics(out["y"], out["l2"].argmax(1))["f1_macro"]
    return s, a


def train_cascade(
    model,
    data,
    teacher_logits=None,
    epochs=15,
    lr=2e-3,
    batch_size=128,
    gamma=0.5,
    delta=0.3,
    tau_kd=3.0,
    pip_warmup_epochs=3,
    seed=42,
    tag="cascade",
):
    """Joint deep-supervised training with eq. 5. Model selection on mean(Sentinel, Analyst) val macro-F1.

    ``pip_warmup_epochs``: keep the PCA-initialised PIP frozen for the first epochs so the randomly
    initialised layers above it do not destroy the information-preserving initialisation.
    """
    torch.manual_seed(seed)
    weights = compute_class_weights(data.y["train"], data.num_classes)
    loss_fn = CascadeLoss(gamma=gamma if teacher_logits is not None else 0.0, delta=delta, tau_kd=tau_kd, class_weights=weights)
    loader = make_loader(data.X["train"], data.y["train"], batch_size, shuffle=True, teacher_logits=teacher_logits, seed=seed)
    opt = torch.optim.AdamW([p for p in model.parameters()], lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    best, best_state, history = -1.0, None, []
    for ep in range(epochs):
        model.train()
        model.freeze_pip(ep < pip_warmup_epochs)
        tot, n = 0.0, 0
        for batch in loader:
            X, y = batch[0], batch[1]
            t = batch[2] if len(batch) > 2 else None
            opt.zero_grad()
            l1, l2 = model(X, mode="train")
            loss = loss_fn(l1, l2, y, t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * len(y)
            n += len(y)
        sched.step()
        s_f1, a_f1 = _val_score(model, data.X["val"], data.y["val"])
        history.append({"epoch": ep + 1, "loss": tot / max(n, 1), "val_f1_sentinel": s_f1, "val_f1_analyst": a_f1})
        logger.info(f"[{tag}] epoch {ep + 1}/{epochs} loss={tot / max(n, 1):.4f} val F1 sentinel={s_f1:.4f} analyst={a_f1:.4f}")
        score = (s_f1 + a_f1) / 2
        if score > best:
            best, best_state = score, copy.deepcopy(model.state_dict())
    model.freeze_pip(False)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


# --------------------------------------------------------------------------------------------- calibration + eval
def calibrate_model(model, data, **gate_kwargs) -> dict:
    out = collect_outputs(model, make_loader(data.X["calib"], data.y["calib"], 512))
    gate = calibrate_arrays(out["l1"], out["l2"], out["y"], **gate_kwargs)
    model.T1.fill_(gate["T1"])
    model.T2.fill_(gate["T2"])
    model.verify_alerts = gate["verify_alerts"]
    return gate


def naive_gate(threshold=0.9) -> dict:
    """Ablation: fixed thresholds, no temperature scaling, no statistical guarantee."""
    return {"T1": 1.0, "T2": 1.0, "tau_b": threshold, "tau_m": threshold, "verify_alerts": False, "feasible": None, "naive": True}


def run_cascade_inference(model, X, gate, batch_size=256) -> dict:
    """Genuine cascade inference over X (Analyst skipped for exited windows). Returns arrays + timing."""
    model.eval()
    preds, probs, exits, n_analyst = [], [], [], 0
    t0 = time.perf_counter()
    for i in range(0, len(X), batch_size):
        out = model.cascade_predict(torch.as_tensor(X[i : i + batch_size], dtype=torch.float32), gate["tau_b"], gate["tau_m"])
        preds.append(out["preds"].numpy())
        probs.append(out["probs"].numpy())
        exits.append(out["exit_mask"].numpy())
        n_analyst += out["n_analyst"]
    elapsed = time.perf_counter() - t0
    k = model.K
    return {
        "preds": np.concatenate(preds) if preds else np.zeros(0, int),
        "probs": np.concatenate(probs) if probs else np.zeros((0, k)),
        "exit_mask": np.concatenate(exits) if exits else np.zeros(0, bool),
        "n_analyst": n_analyst,
        "seconds": elapsed,
    }


def evaluate_model(model, data, gate, split="test") -> dict:
    """Offline evaluation from both heads + a genuine cascade-inference pass that must agree with it."""
    out = collect_outputs(model, make_loader(data.X[split], data.y[split], 512))
    report, preds, _, exit_flags = evaluate_gate_arrays(out["l1"], out["l2"], out["y"], gate, gate.get("verify_alerts", False))
    live = run_cascade_inference(model, data.X[split], gate)
    report["live_inference"] = {
        "agreement_with_offline": float((live["preds"] == preds).mean()) if len(preds) else 1.0,
        "exit_agreement": float((live["exit_mask"] == exit_flags).mean()) if len(preds) else 1.0,
        "analyst_invocations": int(live["n_analyst"]),
        "windows": int(len(preds)),
        "analyst_skipped_fraction": float(1 - live["n_analyst"] / max(len(preds), 1)),
        "seconds_total_batched": float(live["seconds"]),
    }
    return report
