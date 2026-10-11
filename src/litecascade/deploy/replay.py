"""Traffic Replay Engine and Inference Backend for LiteCascade Demo Dashboard (Stage 16).

Streams preprocessed network windows in temporal order, runs early-exit inference via
the exported INT8 TFLite model, enforces calibrated risk gates, supports synthetic attack
injection, and records per-window execution telemetry.
"""

from __future__ import annotations

import argparse
import collections
import datetime
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("litecascade.deploy.replay")

# Profiler MAC constants
SENTINEL_MACS_M = 0.008
ANALYST_MACS_M = 0.221


class ReplayEngine:
    """Streams test windows, runs early-exit inference, and logs real-time metrics."""

    def __init__(
        self,
        tflite_path: str | Path | None = None,
        gate_path: str | Path | None = None,
        dataset_name: str = "fixture",
        max_buffer_size: int = 500,
        seed: int = 42,
    ):
        self.seed = seed
        np.random.seed(seed)
        self.max_buffer_size = max_buffer_size
        self.dataset_name = dataset_name

        # Resolve paths
        self.tflite_path = self._resolve_model_path(tflite_path)
        self.gate_config = self._load_gate_config(gate_path)

        # Initialize model
        self.model_wrapper = None
        self._init_model()

        # Load data windows
        self.is_synthetic = False
        self.X_test, self.y_test, self.families_test = self._load_windows(dataset_name)

        # Attack injection map: index -> family_name
        self.injected_attacks: dict[int, str] = {}

        # Telemetry ring buffer and counters
        self.events: collections.deque[dict[str, Any]] = collections.deque(maxlen=max_buffer_size)
        self.current_index = 0
        self.total_windows = 0
        self.sentinel_exits = 0
        self.analyst_exits = 0
        self.cumulative_macs = 0.0
        self.cumulative_latency_ms = 0.0
        self.correct_predictions = 0
        self.total_alerts = 0

    def _resolve_model_path(self, path: str | Path | None) -> Path | None:
        candidates = [
            Path(path) if path else None,
            Path("checkpoints/litecascade_int8.tflite"),
            Path("checkpoints/litecascade_R_int8.tflite"),
            Path("checkpoints/litecascade_sentinel_int8.tflite"),
        ]
        for c in candidates:
            if c and c.exists():
                return c
        return None

    def _load_gate_config(self, path: str | Path | None) -> dict[str, float]:
        candidates = [
            Path(path) if path else None,
            Path("checkpoints/gate_report.json"),
            Path("checkpoints/gate_m9_stage_verify.json"),
            Path("results/m9_stage_verify/gate_report.json"),
        ]
        for c in candidates:
            if c and c.exists():
                try:
                    with open(c, encoding="utf-8") as f:
                        data = json.load(f)
                    tau_b = float(data.get("tau_b", 0.90))
                    tau_m = float(data.get("tau_m", 0.80))
                    temp = float(data.get("temperature", 1.0))
                    logger.info(f"Loaded gate config from {c}: tau_b={tau_b:.3f}, tau_m={tau_m:.3f}, T={temp:.3f}")
                    return {"tau_b": tau_b, "tau_m": tau_m, "temperature": temp}
                except Exception as e:
                    logger.warning(f"Error reading gate config from {c}: {e}")

        logger.info("Using default gate config: tau_b=0.90, tau_m=0.80, T=1.0")
        return {"tau_b": 0.90, "tau_m": 0.80, "temperature": 1.0}

    def _init_model(self) -> None:
        if self.tflite_path and self.tflite_path.exists():
            try:
                from litecascade.compress.export import TFLiteInterpreterWrapper
                self.model_wrapper = TFLiteInterpreterWrapper(self.tflite_path)
                logger.info(f"Initialized TFLite interpreter wrapper with {self.tflite_path}")
            except Exception as e:
                logger.warning(f"Failed to load TFLite model {self.tflite_path}: {e}")
                self.model_wrapper = None
        else:
            logger.warning("No TFLite model found; operating in mock inference mode.")
            self.model_wrapper = None

    def _load_windows(self, dataset_name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
        from litecascade.data import create_group_splits, create_windows, load_nbaiot

        is_fixture = dataset_name == "fixture"
        df = load_nbaiot(use_synthetic=is_fixture)
        self.is_synthetic = bool(is_fixture or df.get("is_synthetic", False).any())

        splits = create_group_splits(df, dataset_name, seed=self.seed)
        feature_cols = [c for c in df.columns if c not in ["label_binary", "label_family", "group", "timestamp", "is_synthetic"]]

        test_df = df.loc[splits["test"]].copy()
        X_test, y_test = create_windows(test_df, feature_cols, T=10)

        # Extract family labels per window
        families: list[str] = []
        for _, group_df in test_df.groupby("group"):
            group_df = group_df.sort_values("timestamp")
            fam_vals = group_df["label_family"].tolist() if "label_family" in group_df else ["Unknown"] * len(group_df)
            for i in range(0, len(group_df) - 10 + 1):
                families.append(fam_vals[i + 10 - 1])

        if len(families) != len(X_test):
            families = ["Benign" if y == 0 else "Malicious" for y in y_test]

        logger.info(f"Loaded {len(X_test)} test windows (is_synthetic={self.is_synthetic})")
        return X_test, y_test, families

    def inject_attack(self, family: str, start_index: int, duration: int) -> None:
        """Schedules an attack segment for the specified duration starting at start_index."""
        for idx in range(start_index, start_index + duration):
            self.injected_attacks[idx] = family
        logger.info(f"Injected attack {family} from window {start_index} to {start_index + duration - 1}")

    def clear_injected_attacks(self) -> None:
        """Clears all scheduled attack injections."""
        self.injected_attacks.clear()

    def step(self, speed_delay: float = 0.0) -> dict[str, Any]:
        """Processes the next window, executes early-exit inference, and updates telemetry."""
        if speed_delay > 0:
            time.sleep(speed_delay)

        idx = self.current_index
        window = self.X_test[idx : idx + 1]  # Shape (1, 10, F)
        true_label = int(self.y_test[idx])
        family_name = self.families_test[idx]

        # Check for attack injection override
        if idx in self.injected_attacks:
            family_name = self.injected_attacks[idx]
            true_label = 1

        t0 = time.perf_counter()
        sentinel_logits, analyst_logits = self._run_inference(window)
        t_sentinel = time.perf_counter() - t0

        # Apply temperature scaling to Sentinel
        temp = self.gate_config["temperature"]
        scaled_sentinel = sentinel_logits / max(temp, 1e-3)
        exp_s = np.exp(scaled_sentinel - np.max(scaled_sentinel))
        probs_s = exp_s / np.sum(exp_s)
        sentinel_pred = int(np.argmax(probs_s))
        sentinel_conf = float(probs_s[sentinel_pred])

        # Evaluate risk gate
        tau_b = self.gate_config["tau_b"]
        tau_m = self.gate_config["tau_m"]

        exited = False
        if sentinel_pred == 0 and sentinel_conf >= tau_b:
            exited = True
        elif sentinel_pred != 0 and sentinel_conf >= tau_m:
            exited = True

        if exited:
            exit_used = "sentinel"
            final_pred = sentinel_pred
            final_conf = sentinel_conf
            latency_ms = t_sentinel * 1000.0
            macs_used = SENTINEL_MACS_M
        else:
            exit_used = "analyst"
            t1 = time.perf_counter()
            # Temperature scaling for Analyst (default T2=1.0)
            exp_a = np.exp(analyst_logits - np.max(analyst_logits))
            probs_a = exp_a / np.sum(exp_a)
            final_pred = int(np.argmax(probs_a))
            final_conf = float(probs_a[final_pred])
            t_analyst = time.perf_counter() - t1

            latency_ms = (t_sentinel + t_analyst) * 1000.0
            macs_used = SENTINEL_MACS_M + ANALYST_MACS_M

        is_correct = bool(final_pred == true_label)
        is_alert = bool(final_pred == 1)

        event = {
            "index": self.total_windows,
            "window_id": idx,
            "timestamp": datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3],
            "prediction": final_pred,
            "true_label": true_label,
            "confidence": round(final_conf, 4),
            "exit_used": exit_used,
            "latency_ms": round(latency_ms, 3),
            "macs_m": round(macs_used, 4),
            "family": family_name,
            "is_correct": is_correct,
            "is_alert": is_alert,
        }

        # Update telemetry
        self.events.append(event)
        self.total_windows += 1
        if exited:
            self.sentinel_exits += 1
        else:
            self.analyst_exits += 1
        self.cumulative_macs += macs_used
        self.cumulative_latency_ms += latency_ms
        if is_correct:
            self.correct_predictions += 1
        if is_alert:
            self.total_alerts += 1

        # Advance window cursor
        self.current_index = (self.current_index + 1) % len(self.X_test)
        return event

    def _run_inference(self, window: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Runs the model or falls back to synthetic deterministic logits."""
        if self.model_wrapper is not None:
            try:
                s_out, a_out = self.model_wrapper.predict(window)
                return s_out[0], a_out[0]
            except Exception as e:
                logger.warning(f"TFLite forward failed: {e}; using fallback.")

        # Deterministic simulation based on window features
        val = float(np.mean(window))
        if val > 0:
            s_logits = np.array([-1.2, 2.1], dtype=np.float32)
            a_logits = np.array([-2.5, 3.4], dtype=np.float32)
        else:
            s_logits = np.array([2.5, -1.8], dtype=np.float32)
            a_logits = np.array([3.8, -2.9], dtype=np.float32)
        return s_logits, a_logits

    def run(self, seconds: int = 20, speed: float = 1.0) -> dict[str, Any]:
        """Runs replay engine for given number of steps/seconds and returns summary report."""
        t_start = time.perf_counter()
        step_delay = (1.0 / max(speed, 1.0)) * 0.01  # Scaled pacing

        count = 0
        while (time.perf_counter() - t_start) < seconds and count < len(self.X_test):
            self.step(speed_delay=step_delay)
            count += 1

        return self.get_summary()

    def get_summary(self) -> dict[str, Any]:
        """Computes and returns cumulative telemetry metrics."""
        n = max(self.total_windows, 1)
        exit_rate = self.sentinel_exits / n
        accuracy = self.correct_predictions / n
        avg_lat = self.cumulative_latency_ms / n
        avg_macs = self.cumulative_macs / n

        # Expected cost vs analyst baseline
        analyst_macs = ANALYST_MACS_M
        expected_macs = SENTINEL_MACS_M + (1.0 - exit_rate) * ANALYST_MACS_M
        speedup = analyst_macs / max(expected_macs, 1e-6)

        return {
            "total_windows": self.total_windows,
            "sentinel_exits": self.sentinel_exits,
            "analyst_exits": self.analyst_exits,
            "exit_rate": round(exit_rate, 4),
            "accuracy": round(accuracy, 4),
            "total_alerts": self.total_alerts,
            "avg_latency_ms": round(avg_lat, 3),
            "avg_macs_m": round(avg_macs, 4),
            "expected_macs_m": round(expected_macs, 4),
            "speedup_vs_analyst": round(speedup, 2),
            "is_synthetic": self.is_synthetic,
            "dataset_name": self.dataset_name,
        }

    def get_recent_events(self, limit: int = 50) -> list[dict[str, Any]]:
        """Returns the most recent events up to limit."""
        evs = list(self.events)
        return evs[-limit:] if len(evs) > limit else evs


def main() -> None:
    parser = argparse.ArgumentParser(description="LiteCascade Traffic Replay Engine")
    parser.add_argument("--dataset", type=str, default="fixture", help="Dataset name (fixture / real)")
    parser.add_argument("--seconds", type=int, default=20, help="Duration to run replay in seconds")
    parser.add_argument("--speed", type=float, default=10.0, help="Replay speed multiplier (1, 10, 100)")
    parser.add_argument("--out", type=str, default="results/demo_check.json", help="Output summary JSON path")
    parser.add_argument("--inject-family", type=str, default=None, help="Inject attack family (e.g. Mirai)")
    parser.add_argument("--inject-start", type=int, default=10, help="Window index to start attack")
    parser.add_argument("--inject-duration", type=int, default=15, help="Number of attack windows to inject")
    args = parser.parse_args()

    engine = ReplayEngine(dataset_name=args.dataset)

    if args.inject_family:
        engine.inject_attack(args.inject_family, args.inject_start, args.inject_duration)

    print(f"Starting replay engine for {args.seconds}s on {args.dataset} (speed={args.speed}x)...")
    summary = engine.run(seconds=args.seconds, speed=args.speed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[COMPLETE] Replay summary written to {out_path}:")
    print(f"  Total Windows:      {summary['total_windows']}")
    print(f"  Sentinel Exits:     {summary['sentinel_exits']} ({summary['exit_rate']*100:.1f}%)")
    print(f"  Analyst Exits:      {summary['analyst_exits']}")
    print(f"  Accuracy:           {summary['accuracy']*100:.1f}%")
    print(f"  Total Alerts:       {summary['total_alerts']}")
    print(f"  Avg Latency:        {summary['avg_latency_ms']:.3f} ms")
    print(f"  Speedup vs Analyst: {summary['speedup_vs_analyst']:.2f}x")
    print(f"  Synthetic Data:     {summary['is_synthetic']}")


if __name__ == "__main__":
    main()
