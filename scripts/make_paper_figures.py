"""Paper figures generated from the real run outputs (model bundles, reports, calibration data).

    python scripts/make_paper_figures.py --out paper/figures

Writes 300-dpi PNGs plus paper/figures/numbers.json with every number the paper quotes.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from litecascade.web.live import GateLab, ModelZoo  # noqa: E402

BLUE, ORANGE, RED, GREY, INK, GREEN = "#2a78d6", "#d4571f", "#d63e3e", "#9aa4b2", "#1b2430", "#138a5e"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})


def fig_architecture(path):
    """Column-width (3.45 in) block diagram of LiteCascade."""
    fig, ax = plt.subplots(figsize=(3.45, 2.15))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 62)
    ax.axis("off")

    def box(x, y, w, h, text, color, sub=None):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.5", fc="white", ec=color, lw=1.2))
        ax.text(
            x + w / 2, y + h / 2 + (2.2 if sub else 0), text, ha="center", va="center", fontsize=6.6, weight="bold", color=INK
        )
        if sub:
            ax.text(x + w / 2, y + h / 2 - 2.9, sub, ha="center", va="center", fontsize=5.2, color="#444")

    def arrow(a, b, color=GREY, ls="-"):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=7, color=color, lw=0.9, linestyle=ls))

    box(1, 44, 20, 13, "Window", GREY, "T x F records")
    box(26, 44, 20, 13, "PIP layer", BLUE, "z = W(x - mu)")
    box(51, 44, 20, 13, "DS stem", BLUE, "dw k=3 + pw")
    box(76, 44, 22, 13, "Sentinel", BLUE, "DS + GAP + FC")
    ax.add_patch(plt.Polygon([[87, 37], [93, 30], [87, 23], [81, 30]], fc="white", ec="#b8860b", lw=1.2))
    ax.text(87, 30, "gate", ha="center", va="center", fontsize=6, weight="bold")
    box(76, 2, 22, 13, "Early exit", BLUE, "release p1")
    box(36, 2, 28, 13, "Analyst", ORANGE, "DS-TCN, BiGRU, attn")
    for a, b in [((21.6, 50.5), (25.4, 50.5)), ((46.6, 50.5), (50.4, 50.5)), ((71.6, 50.5), (75.4, 50.5))]:
        arrow(a, b)
    arrow((87, 43.4), (87, 37.6))
    arrow((87, 22.4), (87, 15.6), BLUE)
    arrow((81.5, 28), (64.6, 12), ORANGE, ls="--")
    arrow((58, 43.4), (52, 15.6), ORANGE, ls=":")
    ax.text(89, 18.6, "confident", fontsize=5.2, color=BLUE)
    ax.text(66, 24.5, "uncertain", fontsize=5.2, color=ORANGE)
    ax.text(31, 28, "stem features\nreused", fontsize=5.0, color=ORANGE, ha="center")
    fig.tight_layout(pad=0.05)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def fig_pareto(cards, path):
    fig, ax = plt.subplots(figsize=(3.45, 2.5))
    for c in cards:
        m = c["metrics"]
        if m.get("macs") is None or m.get("f1_macro") is None:
            continue
        isc = c["kind"] == "cascade"
        ax.scatter(m["macs"], m["f1_macro"], s=14 + 70 * (m["params"] / 750_000) ** 0.5 * 4, c=BLUE if isc else GREY,
                   edgecolors="white", linewidths=0.6, zorder=3, alpha=0.95 if isc else 0.8)  # fmt: skip
        label = c["name"].replace("LiteCascade", "LC").replace(" (SNIPE)", "").replace("Tiny ", "")
        label = label.replace(" PCA-BiLSTM INT8", " PCA-BiLSTM").replace("Teacher CNN-BiLSTM", "Teacher")
        lead = {"arrowstyle": "-", "color": "#999", "lw": 0.5}
        if c["arch"] == "b9":  # crowded cluster: label in free space with a leader line
            ax.annotate(label, (m["macs"], m["f1_macro"]), xytext=(1.4e6, 0.9405), fontsize=5.4, color="#555", arrowprops=lead)
            continue
        if c["arch"] == "b11":
            ax.annotate(label, (m["macs"], m["f1_macro"]), xytext=(1.1e5, 0.9275), fontsize=5.4, color="#555", arrowprops=lead)
            continue
        off = {"b5": (6, 6), "b6": (6, -1), "teacher": (-14, 6)}.get(c["arch"], (4, 2))
        ax.annotate(label, (m["macs"], m["f1_macro"]), xytext=off, textcoords="offset points", fontsize=5.4,
                    color=INK if isc else "#555")  # fmt: skip
    ax.set_xscale("log")
    ax.set_xlim(4e4, 1.6e7)
    ax.set_xlabel("MACs per window (LiteCascade: expected E[C])")
    ax.set_ylabel("Test macro-F1")
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout(pad=0.2)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def fig_gate(curve, deployed_eps, path):
    eps = [p["epsilon"] for p in curve]
    fig, ax = plt.subplots(figsize=(3.45, 2.3))
    ax.plot(eps, [p["rho"] for p in curve], "-o", ms=2.5, color=BLUE, label="exit rate rho (test)")
    ax.plot(eps, [p["f1_macro"] for p in curve], "-", color=INK, lw=1, label="cascade macro-F1 (test)")
    ax.set_xscale("log")
    ax.set_xlabel("risk budget epsilon")
    ax.set_ylim(0, 1.02)
    ax2 = ax.twinx()
    ax2.plot(eps, [100 * p["exit_induced_miss_rate"] for p in curve], "-s", ms=2.5, color=RED, label="exit-induced miss % (test)")
    ax2.plot(eps, [100 * e for e in eps], ":", color=RED, lw=1, label="bound epsilon")
    ax2.set_ylim(0, 10.5)
    ax2.set_ylabel("miss rate (%)", color=RED)
    ax2.spines["right"].set_visible(True)
    infeasible = [p["epsilon"] for p in curve if not p["feasible"]]
    if infeasible:
        ax.axvspan(min(eps) * 0.9, max(infeasible) * 1.15, color="#b8860b", alpha=0.12, lw=0)
    ax.axvline(deployed_eps, color=GREY, lw=0.8, ls="--")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=5.6, loc="center right", frameon=False)
    fig.tight_layout(pad=0.2)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def fig_stages(stages, path):
    names = ["Joint+KD", "Pruned 30%", "INT8 (old gate)", "INT8 recal."]
    f1 = [s["test"]["cascade"]["f1_macro"] for s in stages]
    rho = [s["test"]["rho_raw"] for s in stages]
    size = [s["profile"]["size_kb_serialized"] for s in stages]
    fig, ax = plt.subplots(figsize=(3.45, 2.2))
    x = range(len(stages))
    ax.bar([i - 0.2 for i in x], size, width=0.38, color=GREY, label="size (KB)")
    for i, v in enumerate(size):
        ax.text(i - 0.2, v + 1.5, f"{v:.0f}", ha="center", fontsize=6)
    ax.set_ylabel("serialized size (KB)")
    ax.set_xticks(list(x), names[: len(stages)], fontsize=6.5)
    ax2 = ax.twinx()
    ax2.plot(list(x), f1, "-o", color=INK, ms=3, label="macro-F1")
    ax2.plot(list(x), rho, "-s", color=BLUE, ms=3, label="exit rate rho")
    ax2.set_ylim(0.5, 1.0)
    ax2.spines["right"].set_visible(True)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=6, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    fig.tight_layout(pad=0.2)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/figures")
    ap.add_argument("--report", default="reports/fixture.json")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    zoo = ModelZoo("artifacts")
    cards = [c for c in zoo.cards() if c["id"] != "fixture_R_fp32"]
    rep = json.loads(Path(args.report).read_text())
    casc = rep["cascades"]
    from litecascade.data.pipeline import prepare_data

    prim = zoo.bundles["fixture_R_int8"]
    data = prepare_data("fixture", T=prim.window)
    lab = GateLab(prim, data)
    curve = lab.curve(0.05)

    fig_architecture(out / "fig1_architecture.png")
    fig_pareto(cards, out / "fig2_pareto.png")
    fig_gate(curve, prim.gate["epsilon"], out / "fig3_gate.png")
    fig_stages(casc["R"]["stages"], out / "fig4_stages.png")

    numbers = {
        "cards": cards,
        "gate_curve": curve,
        "stages_R": [
            {"stage": s["stage"], "test": s["test"], "profile": s["profile"], "gate": s["gate"]} for s in casc["R"]["stages"]
        ],
        "stages_T": [{"stage": s["stage"], "test": s["test"], "profile": s["profile"]} for s in casc["T"]["stages"]],
        "data": casc["R"]["data"],
        "teacher": casc["R"]["teacher"],
        "ablations": rep.get("ablations"),
        "baselines": rep["baselines"],
    }
    (out / "numbers.json").write_text(json.dumps(numbers, indent=1, default=float))
    print(f"Figures + numbers written to {out}")


if __name__ == "__main__":
    main()
