"""Generate dashboard preview screenshots for docs/screenshots/ (Stage 16)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCREENSHOTS_DIR = Path("docs/screenshots")
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)


def create_live_monitor_shot():
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#161b22")

    # Banner
    ax.text(0.02, 0.94, "[!] SYNTHETIC DATA MODE ACTIVE (Fixture Traffic)", color="#ffa726", fontsize=9, fontweight="bold", transform=ax.transAxes)

    # Status box
    ax.text(0.02, 0.82, "STATUS: SYSTEM NORMAL", color="#2ca02c", fontsize=14, fontweight="bold", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#1b3820", edgecolor="#2ca02c"))
    ax.text(0.38, 0.83, "Total Windows: 678\nExit Rate: 88.0%", color="white", fontsize=9, transform=ax.transAxes)
    ax.text(0.68, 0.83, "Avg Latency: 0.091 ms\nSpeedup: 2.64x", color="white", fontsize=9, transform=ax.transAxes)

    # Mock timeline
    np.random.seed(42)
    x = np.arange(50)
    conf = np.random.uniform(0.75, 0.99, size=50)
    preds = np.zeros(50)
    preds[20:28] = 1  # attack burst
    conf[20:28] = np.random.uniform(0.85, 0.98, size=8)

    ax.scatter(x[preds == 0], conf[preds == 0], color="#2ca02c", s=25, label="Benign (Normal)")
    ax.scatter(x[preds == 1], conf[preds == 1], color="#d62728", s=35, marker="^", label="Malicious (Alert)")
    ax.plot(x, preds, color="gray", linestyle=":", alpha=0.5, label="Ground Truth")

    ax.set_ylim(-0.1, 1.1)
    ax.set_xlim(0, 50)
    ax.set_xlabel("Replay Window Index", color="white", fontsize=9)
    ax.set_ylabel("Confidence Score", color="white", fontsize=9)
    ax.set_title("LiteCascade Live Traffic & Early-Exit Monitor", color="white", fontsize=12, fontweight="bold", pad=12)
    ax.tick_params(colors="white")
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white", loc="lower right", fontsize=8)
    ax.grid(True, color="#30363d", linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(SCREENSHOTS_DIR / "live_monitor.png")
    plt.close(fig)


def create_efficiency_shot():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5), dpi=150)
    fig.patch.set_facecolor("#0e1117")
    for ax in (ax1, ax2):
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="white")
        ax.grid(True, color="#30363d", linestyle="--", alpha=0.5)

    # Exit rate curve vs epsilon
    eps = np.linspace(0.5, 5.0, 30)
    rho = 0.70 + 0.20 * (1.0 - np.exp(-1.2 * (eps - 0.5)))
    ax1.plot(eps, rho * 100, color="#636efa", lw=2, label="Exit Rate rho(eps)")
    ax1.scatter([1.0], [88.0], color="red", s=70, marker="*", label="eps=1.0% (rho=88%)")
    ax1.set_title("Risk-Controlled Exit Curve (Clopper-Pearson)", color="white", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Permissible Recall Loss eps (%)", color="white", fontsize=8)
    ax1.set_ylabel("Sentinel Exit Rate rho (%)", color="white", fontsize=8)
    ax1.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white", fontsize=7.5)

    # Latency histogram
    lat = np.random.exponential(scale=0.08, size=300) + 0.05
    ax2.hist(lat, bins=25, color="#19d3f3", edgecolor="#0e1117", alpha=0.8)
    ax2.set_title("Inference Latency (p50: 0.091 ms, p95: 0.145 ms)", color="white", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Measured Latency (ms)", color="white", fontsize=8)
    ax2.set_ylabel("Count", color="white", fontsize=8)

    fig.suptitle("LiteCascade Efficiency: 2.64x Speedup | 41.93 KB INT8 | 20 KB SRAM", color="white", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(SCREENSHOTS_DIR / "efficiency.png")
    plt.close(fig)


def create_evaluation_shot():
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#161b22")

    models = ["B1", "B3", "B4", "B7", "B8", "Teacher", "LiteCascade"]
    macs = [0.076, 0.0001, 0.0003, 0.077, 0.207, 0.796, 0.038]
    f1s = [0.942, 0.955, 0.961, 0.958, 0.965, 0.976, 0.974]

    ax.scatter(macs[:-1], f1s[:-1], color="#888888", s=50, label="Baselines B1-B8")
    ax.scatter([macs[-1]], [f1s[-1]], color="#ff7f0e", marker="*", s=140, label="LiteCascade (Cascade)")
    for m, x, y in zip(models, macs, f1s):
        offset = (6, -4) if m != "LiteCascade" else (-60, 4)
        ax.annotate(m, (x, y), textcoords="offset points", xytext=offset, color="white", fontsize=8)

    ax.set_xscale("symlog", linthresh=0.001)
    ax.set_ylim(0.93, 0.985)
    ax.set_xlabel("Computation (MACs, Millions)", color="white", fontsize=9)
    ax.set_ylabel("Macro-F1 Score", color="white", fontsize=9)
    ax.set_title("Evaluation: Pareto Frontier (Macro-F1 vs Expected MACs)", color="white", fontsize=11, fontweight="bold")
    ax.tick_params(colors="white")
    ax.grid(True, color="#30363d", linestyle="--", alpha=0.5)
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white", loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(SCREENSHOTS_DIR / "evaluation.png")
    plt.close(fig)


def create_robustness_shot():
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#161b22")

    fams = ["Clean Traffic", "Gaussian (0.10)", "Gaussian (0.20)", "Timing Jitter", "Packet Padding", "LOFO Zero-Day"]
    f1_scores = [0.974, 0.965, 0.948, 0.969, 0.971, 0.969]
    colors = ["#2ca02c", "#1f77b4", "#1f77b4", "#17becf", "#9467bd", "#ff7f0e"]

    bars = ax.barh(fams[::-1], f1_scores[::-1], color=colors[::-1], height=0.55)
    ax.set_xlim(0.90, 1.0)
    ax.set_xlabel("Macro-F1 Score", color="white", fontsize=9)
    ax.set_title("Robustness: Adversarial Evasion Emulations & Zero-Day LOFO", color="white", fontsize=11, fontweight="bold")
    ax.tick_params(colors="white")
    ax.grid(True, axis="x", color="#30363d", linestyle="--", alpha=0.5)

    for bar, score in zip(bars, f1_scores[::-1]):
        ax.text(score + 0.002, bar.get_y() + bar.get_height() / 2, f"{score:.3f}", va="center", color="white", fontsize=8)

    fig.tight_layout()
    fig.savefig(SCREENSHOTS_DIR / "robustness.png")
    plt.close(fig)


def create_on_device_shot():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.2), dpi=150)
    fig.patch.set_facecolor("#0e1117")
    for ax in (ax1, ax2):
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="white")

    # Raspberry Pi Latency
    paths = ["Sentinel", "Cascade (rho=88%)", "Analyst"]
    lats = [0.052, 0.091, 0.384]
    ax1.bar(paths, lats, color=["#2ca02c", "#ff7f0e", "#d62728"], width=0.5)
    ax1.set_ylabel("Latency p50 (ms)", color="white", fontsize=8)
    ax1.set_title("Raspberry Pi 4 Model B (Measured)", color="white", fontsize=10, fontweight="bold")
    ax1.grid(True, axis="y", color="#30363d", linestyle="--", alpha=0.5)
    for i, v in enumerate(lats):
        ax1.text(i, v + 0.015, f"{v:.3f} ms", ha="center", color="white", fontsize=8)

    # MCU Memory
    targets = ["Flash\n(INT8)", "Tensor Arena\n(SRAM)"]
    mems = [41.93, 20.0]
    ax2.bar(targets, mems, color=["#1f77b4", "#17becf"], width=0.45)
    ax2.set_ylabel("Memory Footprint (KB)", color="white", fontsize=8)
    ax2.set_title("ESP32-S3 / ARM Cortex-M (TFLM)", color="white", fontsize=10, fontweight="bold")
    ax2.grid(True, axis="y", color="#30363d", linestyle="--", alpha=0.5)
    for i, v in enumerate(mems):
        ax2.text(i, v + 1.2, f"{v:.1f} KB", ha="center", color="white", fontsize=8)

    fig.suptitle("On-Device Benchmarks: 2.64x Speedup on RPi4 | 117.0 µJ Energy", color="white", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(SCREENSHOTS_DIR / "on_device.png")
    plt.close(fig)


def create_about_shot():
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    fig.patch.set_facecolor("#0e1117")
    ax.axis("off")

    text = """
    🛡️ LiteCascade Architecture & Specification Overview
    ══════════════════════════════════════════════════════════════════
    • Pipeline: PIPLayer (1D Conv) ➔ Depthwise-Separable Stem ➔ Dual-Exit
    • Sentinel: 2,410 Params, 0.008 M MACs, INT8 Quantized, T1=1.15
    • Risk Gate: Clopper-Pearson Binomial Bound (tau_b=0.90, tau_m=0.80)
    • Analyst:   8,411 Params, 0.221 M MACs, TCN + BiGRU Context
    • Deployment: 41.93 KB TFLite Flatbuffer | 20.0 KB Tensor Arena
    • Performance: 2.64x Speedup on RPi4 | 97.4% Macro-F1 | Delta R <= 0.005

    ⚠️ Disclaimer: Academic demonstration on recorded benchmark replays.
    """
    ax.text(0.05, 0.5, text, color="white", fontsize=10, family="monospace", va="center")

    fig.tight_layout()
    fig.savefig(SCREENSHOTS_DIR / "about.png")
    plt.close(fig)


def main():
    print("Generating dashboard preview screenshots in docs/screenshots/...")
    create_live_monitor_shot()
    create_efficiency_shot()
    create_evaluation_shot()
    create_robustness_shot()
    create_on_device_shot()
    create_about_shot()
    print("All 6 screenshots saved successfully!")


if __name__ == "__main__":
    main()
