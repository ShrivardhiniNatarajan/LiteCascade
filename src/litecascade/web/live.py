"""Model zoo + live replay engine behind the dashboard.

The replay streams REAL held-out test windows (devices never seen in training) through every loaded model.
Every prediction comes from the model itself: ground-truth labels are only used afterwards to score the
predictions (running accuracy / F1), never to produce them. Latencies are measured per call on this CPU.
"""

import math
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from litecascade.data.pipeline import prepare_data
from litecascade.deploy.bundle import Bundle, list_bundles
from litecascade.eval.gate import collect_outputs, select_tau_b, softmax_t
from litecascade.eval.metrics import compute_detection_metrics

FAMILY_ORDER = ["cascade", "hybrid", "recurrent", "convolutional", "attention", "mlp"]


def _f(v):
    return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else v


class ModelZoo:
    """All bundles under ``root`` that share a dataset and feature set."""

    def __init__(self, root="artifacts", dataset: str | None = None):
        self.root = Path(root)
        self.bundles: dict[str, Bundle] = {}
        for p in list_bundles(self.root):
            try:
                b = Bundle(p)
            except Exception as e:  # a broken bundle must not take the dashboard down
                print(f"[zoo] skipping {p}: {e}")
                continue
            if dataset and b.meta.get("dataset") != dataset:
                continue
            self.bundles[b.name] = b
        if not self.bundles:
            raise FileNotFoundError(f"No model bundles under '{self.root}'. Run scripts/run_demo.py first.")
        ref = self.primary()
        self.bundles = {k: b for k, b in self.bundles.items() if b.feature_cols == ref.feature_cols and b.window == ref.window}

    def primary(self) -> Bundle:
        """Preferred LiteCascade bundle: INT8 R > FP32 R > any cascade > anything."""
        order = sorted(
            self.bundles.values(),
            key=lambda b: (
                b.kind != "cascade",
                b.kind == "cascade" and b.meta["model_config"]["variant"] != "R",
                b.meta.get("quantization") == "none",
                b.name,
            ),
        )
        return order[0]

    def card(self, b: Bundle) -> dict:
        m = b.meta.get("metrics", {})
        if b.kind == "cascade":
            t, prof = m.get("test", {}), m.get("profile", {})
            c = t.get("cascade", {})
            lat = prof.get("latency_full_path", {}).get("p50_ms")
            metrics = {
                "f1_macro": c.get("f1_macro"),
                "recall_malicious": c.get("recall_malicious"),
                "fpr": c.get("fpr"),
                "mcc": c.get("mcc"),
                "params": (prof.get("params") or {}).get("total"),
                "macs": prof.get("expected_macs"),
                "macs_full": prof.get("macs_full"),
                "macs_sentinel": prof.get("macs_sentinel_path"),
                "size_kb": prof.get("size_kb_serialized"),
                "latency_p50_ms": lat,
                "latency_sentinel_ms": prof.get("latency_sentinel_path", {}).get("p50_ms"),
                "rho": t.get("rho_raw"),
                "exit_induced_miss_rate": t.get("exit_induced_miss_rate"),
                "sentinel_f1": t.get("sentinel_only", {}).get("f1_macro"),
                "analyst_f1": t.get("analyst_only", {}).get("f1_macro"),
                "measured_speedup": prof.get("measured_speedup"),
                "param_breakdown": prof.get("params"),
            }
        else:
            size = m.get("size_int8_kb") if b.meta.get("quantization") != "none" else m.get("size_fp32_kb")
            metrics = {
                "f1_macro": m.get("f1_macro"),
                "f1_ci": m.get("f1_macro_95ci"),
                "recall_malicious": m.get("recall_malicious"),
                "fpr": m.get("fpr"),
                "mcc": m.get("mcc"),
                "params": m.get("params"),
                "macs": m.get("macs"),
                "size_kb": size,
                "size_fp32_kb": m.get("size_fp32_kb"),
                "latency_p50_ms": m.get("latency_p50_ms"),
            }
        return {
            "id": b.name,
            "name": b.display_name,
            "kind": b.kind,
            "arch": b.arch,
            "family": b.family,
            "description": b.description,
            "quantization": b.meta.get("quantization", "none"),
            "dataset": b.meta.get("dataset"),
            "is_synthetic": bool(b.meta.get("is_synthetic")),
            "window": b.window,
            "n_features": len(b.feature_cols),
            "class_names": b.class_names,
            "gate": b.gate,
            "created": b.meta.get("created"),
            "metrics": {k: _f(v) for k, v in metrics.items()},
        }

    def cards(self) -> list[dict]:
        cards = [self.card(b) for b in self.bundles.values()]
        return sorted(cards, key=lambda c: (FAMILY_ORDER.index(c["family"]) if c["family"] in FAMILY_ORDER else 99, c["name"]))


class _Stats:
    def __init__(self):
        self.tp = self.fp = self.fn = self.tn = 0
        self.lat = deque(maxlen=120)
        self.lat_total = 0.0
        self.n = 0

    def update(self, y: int, pred: int, ms: float):
        p = int(pred > 0)
        self.tp += y and p
        self.fn += y and not p
        self.fp += (not y) and p
        self.tn += (not y) and not p
        self.lat.append(ms)
        self.lat_total += ms
        self.n += 1

    def as_dict(self) -> dict:
        n = max(self.n, 1)
        prec = self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0
        rec = self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0
        f1m = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        precb = self.tn / (self.tn + self.fn) if self.tn + self.fn else 0.0
        recb = self.tn / (self.tn + self.fp) if self.tn + self.fp else 0.0
        f1b = 2 * precb * recb / (precb + recb) if precb + recb else 0.0
        classes = (self.tp + self.fn > 0) + (self.tn + self.fp > 0)
        return {
            "n": self.n,
            "accuracy": (self.tp + self.tn) / n,
            "f1_macro": (f1m * (self.tp + self.fn > 0) + f1b * (self.tn + self.fp > 0)) / max(classes, 1),
            "recall": rec,
            "fpr": self.fp / (self.fp + self.tn) if self.fp + self.tn else 0.0,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "latency_ms_recent": float(np.mean(self.lat)) if self.lat else None,
            "latency_ms_mean": self.lat_total / n if self.n else None,
            "latency_spark": [round(v, 4) for v in list(self.lat)[-40:]],
        }


class ReplayEngine:
    """Replays held-out test windows (benign stretches with attack bursts) through every model in the zoo."""

    def __init__(self, zoo: ModelZoo, seed: int = 7):
        self.zoo = zoo
        self.primary = zoo.primary()
        meta = self.primary.meta
        self.data = prepare_data(meta["dataset"], T=self.primary.window, task=meta.get("task", "binary"))
        # raw (unscaled) windows so each bundle applies its own train-fitted scaler
        X = self.data.X["test"]
        sc = self.data.preprocessor.scaler
        self.raw = (X * sc.scale_.astype(np.float32) + sc.center_.astype(np.float32)).astype(np.float32)
        self.y = self.data.y["test"]
        self.meta = self.data.meta["test"].reset_index(drop=True)
        self.families = sorted({f for f in self.meta["label_family"].unique() if f != "benign"})
        self.rng = np.random.default_rng(seed)
        # contiguous runs of windows per family (one per segment) -> realistic replay order
        self.runs: dict[str, list[np.ndarray]] = {}
        for (fam, _seg), g in self.meta.groupby(["label_family", "segment"], sort=False):
            self.runs.setdefault(fam, []).append(g.index.to_numpy())
        self.cursor = {f: (0, 0) for f in self.runs}
        self.lock = threading.Lock()
        self.active = list(zoo.bundles)
        self.reset()

    # ------------------------------------------------------------------ schedule
    def _take(self, fam: str, n: int) -> list[int]:
        out = []
        run_i, pos = self.cursor[fam]
        while len(out) < n:
            run = self.runs[fam][run_i % len(self.runs[fam])]
            take = run[pos : pos + n - len(out)]
            out.extend(int(i) for i in take)
            pos += len(take)
            if pos >= len(run):
                run_i, pos = run_i + 1, 0
        self.cursor[fam] = (run_i, pos)
        return out

    def _refill(self):
        if self.rng.random() < 0.55 or not self.families:
            self.queue.extend(("benign", i) for i in self._take("benign", int(self.rng.integers(25, 60))))
        else:
            fam = self.families[int(self.rng.integers(len(self.families)))]
            self.queue.extend((fam, i) for i in self._take(fam, int(self.rng.integers(15, 35))))

    def inject(self, family: str, n: int = 30) -> dict:
        if family not in self.runs:
            raise ValueError(f"unknown family {family}; available: {self.families}")
        with self.lock:
            injected = [(family, i) for i in self._take(family, n)]
            self.queue = deque(injected + list(self.queue))
        return {"family": family, "windows": n}

    def reset(self):
        with self.lock:
            self.queue: deque = deque()
            self.step_no = 0
            self.stats = {k: _Stats() for k in self.zoo.bundles}
            self.cascade_exits = 0
            self.cascade_steps = 0
            self.macs_used = 0.0
            self.macs_full = 0.0
            self.timeline: deque = deque(maxlen=400)
            self.alerts = 0

    # ------------------------------------------------------------------ step
    def step(self) -> dict:
        with self.lock:
            if not self.queue:
                self._refill()
            fam, idx = self.queue.popleft()
            self.step_no += 1
            y = int(self.y[idx] > 0)
            raw = self.raw[idx : idx + 1]
            outputs = {}
            for name in self.active:
                b = self.zoo.bundles[name]
                x = b.preprocessor.scale_array(raw)
                t0 = time.perf_counter()
                out = b.predict_windows(x, batch_size=1)
                ms = (time.perf_counter() - t0) * 1000
                pred = int(out["preds"][0])
                o = {"pred": pred, "score": float(1 - out["probs"][0][0]), "latency_ms": ms}
                if b.kind == "cascade":
                    o["exit"] = bool(out["exit_mask"][0])
                    o["sentinel_benign"] = float(out["sentinel_probs"][0][0])
                    o["sentinel_score"] = float(1 - out["sentinel_probs"][0][0])
                self.stats[name].update(y, pred, ms)
                outputs[name] = o

            prim = outputs.get(self.primary.name)
            prof = self.zoo.card(self.primary)["metrics"]
            if prim is not None:
                self.cascade_steps += 1
                self.cascade_exits += prim["exit"]
                c1 = prof.get("macs_sentinel") or 0
                full = prof.get("macs_full") or 0
                self.macs_used += c1 if prim["exit"] else full
                self.macs_full += full
                self.alerts += prim["pred"] > 0
            event = {
                "step": self.step_no,
                "window_index": idx,
                "segment": str(self.meta.at[idx, "segment"]),
                "family": fam,
                "label": y,
                "primary": self.primary.name,
                "outputs": outputs,
                "ts": time.time(),
            }
            self.timeline.append({"step": self.step_no, "label": y, "family": fam, **({"primary": prim} if prim else {})})
            return event

    def summary(self) -> dict:
        with self.lock:
            per_model = {k: s.as_dict() for k, s in self.stats.items()}
            n = max(self.cascade_steps, 1)
            return {
                "steps": self.step_no,
                "primary": self.primary.name,
                "exit_rate": self.cascade_exits / n if self.cascade_steps else None,
                "analyst_calls": self.cascade_steps - self.cascade_exits,
                "compute_saved": 1 - self.macs_used / self.macs_full if self.macs_full else None,
                "alerts": self.alerts,
                "models": per_model,
                "queue": len(self.queue),
                "families": self.families,
            }


class GateLab:
    """Risk/efficiency trade-off of a cascade bundle, recomputed from its real calibration outputs."""

    def __init__(self, bundle: Bundle, data):
        from litecascade.train.cascade_trainer import make_loader

        self.b, self.data = bundle, data

        def scaled(split):  # re-scale with the bundle's own train-fitted scaler
            sc = data.preprocessor.scaler
            raw = data.X[split] * sc.scale_.astype(np.float32) + sc.center_.astype(np.float32)
            return bundle.preprocessor.scale_array(raw)

        cal = collect_outputs(bundle.model, make_loader(scaled("calib"), data.y["calib"], 512))
        test = collect_outputs(bundle.model, make_loader(scaled("test"), data.y["test"], 512))
        g = bundle.gate
        self.cal_p1 = softmax_t(cal["l1"], g["T1"])
        self.cal_a = cal["l2"].argmax(1)
        self.cal_y = cal["y"]
        self.test_p1 = softmax_t(test["l1"], g["T1"])
        self.test_p2 = softmax_t(test["l2"], g["T2"])
        self.test_y = test["y"]

    def point(self, epsilon: float, alpha: float = 0.05, tau_m: float | None = None) -> dict:
        tb = select_tau_b(self.cal_p1, self.cal_a, self.cal_y, epsilon=epsilon, alpha=alpha)
        tau_m = self.b.gate["tau_m"] if tau_m is None else tau_m
        p1 = self.test_p1
        pred1 = p1.argmax(1)
        exit_b = (pred1 == 0) & (p1[:, 0] >= np.float32(tb["tau_b"]))
        exit_m = (pred1 != 0) & (p1.max(1) >= np.float32(tau_m))
        exits = exit_b | exit_m
        final = np.where(exits[:, None], p1, self.test_p2).argmax(1)
        mal = self.test_y > 0
        miss = (exit_b & mal & (self.test_p2.argmax(1) != 0)).sum() / max(mal.sum(), 1)
        m = compute_detection_metrics(self.test_y, final)
        return {
            "epsilon": epsilon,
            "alpha": alpha,
            "tau_b": tb["tau_b"],
            "feasible": tb["feasible"],
            "N_m": tb["N_m"],
            "min_N_m": tb["min_N_m"],
            "m_max": tb["m_max"],
            "rho": float(exits.mean()),
            "exit_induced_miss_rate": float(miss),
            "f1_macro": m["f1_macro"],
            "recall": m["recall_malicious"],
            "fpr": m["fpr"],
        }

    def curve(self, alpha: float = 0.05) -> list[dict]:
        eps = [0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.1]
        return [self.point(e, alpha) for e in eps]
