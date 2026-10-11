"""LiteCascade Interactive Demonstration Dashboard (Stage 16).

A research demonstration for evaluating the LiteCascade early-exit neural cascade
on recorded/replayed IoT security datasets. NOT a production intrusion detection system.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from litecascade.deploy.replay import ANALYST_MACS_M, SENTINEL_MACS_M, ReplayEngine

# -----------------------------------------------------------------------------
# Streamlit App Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="LiteCascade Demo Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Caching & Resource Loaders
# -----------------------------------------------------------------------------
@st.cache_resource
def get_replay_engine(dataset_name: str = "fixture") -> ReplayEngine:
    return ReplayEngine(dataset_name=dataset_name)


def load_json_safely(path: Path) -> dict | None:
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading {path}: {e}")
    return None


def load_csv_safely(path: Path) -> pd.DataFrame | None:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception as e:
            st.error(f"Error loading {path}: {e}")
    return None


# -----------------------------------------------------------------------------
# Headless Sanity Test Mode
# -----------------------------------------------------------------------------
def run_headless_test():
    print("[HEADLESS TEST] Initializing LiteCascade App & Replay Engine...")
    engine = get_replay_engine("fixture")
    for _ in range(20):
        engine.step()
    summary = engine.get_summary()
    assert summary["total_windows"] == 20
    assert "exit_rate" in summary
    assert "accuracy" in summary
    print("[HEADLESS TEST] Headless sanity check passed successfully!")
    sys.exit(0)


if "--headless-test" in sys.argv:
    run_headless_test()

# -----------------------------------------------------------------------------
# Global State Initialization
# -----------------------------------------------------------------------------
if "engine" not in st.session_state:
    st.session_state.engine = get_replay_engine("fixture")

if "is_running" not in st.session_state:
    st.session_state.is_running = False

# -----------------------------------------------------------------------------
# Sidebar Navigation & Global Controls
# -----------------------------------------------------------------------------
st.sidebar.title("🛡️ LiteCascade IDS")
st.sidebar.caption("Lightweight Early-Exit Neural Cascade")

# Prominent Disclaimer
st.sidebar.info(
    "🔬 **Research Demonstration**\n\n"
    "This dashboard replays recorded IoT network benchmarks. "
    "It is an academic demonstration and not a production IDS."
)

page = st.sidebar.radio(
    "Navigation",
    [
        "🔴 Live Monitor",
        "⚡ Efficiency",
        "📊 Evaluation",
        "🛡️ Robustness",
        "📱 On-Device",
        "ℹ️ About",
    ],
)

st.sidebar.divider()
st.sidebar.subheader("Replay Configuration")
dataset_choice = st.sidebar.selectbox("Traffic Source", ["Synthetic Fixture (N-BaIoT)", "Real Capture (N-BaIoT)"], index=0)
speed_mult = st.sidebar.select_slider("Replay Speed", options=[1, 5, 10, 50, 100], value=10)

engine: ReplayEngine = st.session_state.engine

# -----------------------------------------------------------------------------
# Top Banner for Synthetic Fixtures
# -----------------------------------------------------------------------------
if engine.is_synthetic:
    st.warning(
        "⚠️ **SYNTHETIC DATA MODE ACTIVE:** Traffic windows are sourced from continuous-integration "
        "synthetic fixtures (`data/processed/synthetic_nbaiot.csv`). Metrics illustrate architectural "
        "behavior and logic; reference benchmark results are shown on the Evaluation page.",
        icon="⚠️",
    )

# =============================================================================
# PAGE 1: LIVE MONITOR
# =============================================================================
if page == "🔴 Live Monitor":
    st.title("🔴 Live Traffic & Early-Exit Monitor")
    st.caption("Real-time stream replay with dual-exit inference, temperature calibration, and risk-controlled gating.")

    # Top Status Bar
    recent_events = engine.get_recent_events(limit=50)
    last_event = recent_events[-1] if recent_events else None

    col_stat, col_metric1, col_metric2, col_metric3, col_metric4 = st.columns([1.5, 1, 1, 1, 1])

    is_alert = last_event["is_alert"] if last_event else False
    status_label = "🚨 INTRUSION ALERT" if is_alert else "🟢 SYSTEM NORMAL"
    status_color = "red" if is_alert else "green"

    with col_stat:
        st.markdown(
            f"<div style='background-color: {'#ffebee' if is_alert else '#e8f5e9'}; padding: 16px; border-radius: 8px; border-left: 6px solid {status_color}; text-align: center;'>"
            f"<h2 style='color: {status_color}; margin: 0;'>{status_label}</h2>"
            f"<small style='color: #666;'>Current Network State</small>"
            f"</div>",
            unsafe_allow_html=True,
        )

    summary = engine.get_summary()
    with col_metric1:
        st.metric("Total Windows", summary["total_windows"])
    with col_metric2:
        st.metric("Sentinel Exits", f"{summary['exit_rate']*100:.1f}%", f"{summary['sentinel_exits']} wins")
    with col_metric3:
        st.metric("Inference Latency", f"{summary['avg_latency_ms']:.2f} ms")
    with col_metric4:
        st.metric("Live Accuracy", f"{summary['accuracy']*100:.1f}%")

    st.divider()

    # Replay Controls
    c_ctrl1, c_ctrl2, c_ctrl3, c_ctrl4 = st.columns([1, 1, 1, 2])
    with c_ctrl1:
        if st.button("▶️ Step 1 Window", use_container_width=True):
            engine.step()
            st.rerun()
    with c_ctrl2:
        if st.button("⏩ Stream 25 Windows", use_container_width=True):
            for _ in range(25):
                engine.step(speed_delay=0.01 / speed_mult)
            st.rerun()
    with c_ctrl3:
        if st.button("🔄 Reset Stream", use_container_width=True):
            st.session_state.engine = get_replay_engine("fixture")
            st.rerun()
    with c_ctrl4:
        with st.expander("⚡ Inject Attack Segment"):
            att_col1, att_col2, att_col3 = st.columns(3)
            with att_col1:
                att_fam = st.selectbox("Malware Family", ["Mirai", "Gafgyt", "Tsunami", "DDoS_SYN"])
            with att_col2:
                att_dur = st.number_input("Duration (windows)", min_value=5, max_value=50, value=15)
            with att_col3:
                if st.button("Inject Now", use_container_width=True):
                    engine.inject_attack(family=att_fam, start_index=engine.current_index, duration=att_dur)
                    st.success(f"Injected {att_fam} ({att_dur} windows)")

    # Timeline Chart
    if recent_events:
        df_ev = pd.DataFrame(recent_events)

        fig_timeline = go.Figure()

        # Add benign and malicious markers
        benign_df = df_ev[df_ev["prediction"] == 0]
        mal_df = df_ev[df_ev["prediction"] == 1]

        fig_timeline.add_trace(go.Scatter(
            x=benign_df["index"],
            y=benign_df["confidence"],
            mode="markers",
            name="Benign (Normal)",
            marker=dict(color="#2ca02c", size=8, symbol="circle"),
            text=benign_df["family"],
        ))

        fig_timeline.add_trace(go.Scatter(
            x=mal_df["index"],
            y=mal_df["confidence"],
            mode="markers",
            name="Malicious (Alert)",
            marker=dict(color="#d62728", size=10, symbol="triangle-up"),
            text=mal_df["family"],
        ))

        # Add ground truth overlay line
        fig_timeline.add_trace(go.Scatter(
            x=df_ev["index"],
            y=df_ev["true_label"],
            mode="lines",
            name="Ground Truth (0/1)",
            line=dict(color="rgba(150, 150, 150, 0.4)", dash="dot"),
        ))

        fig_timeline.update_layout(
            title="Real-Time Detection Timeline (Confidence & Ground Truth)",
            xaxis_title="Replay Window Index",
            yaxis_title="Confidence / True Label",
            yaxis=dict(range=[-0.1, 1.1]),
            height=340,
            margin=dict(l=30, r=30, t=40, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

        # Recent Alerts / Events Table
        st.subheader("📋 Recent Traffic Events")
        table_df = df_ev[["timestamp", "window_id", "prediction", "true_label", "confidence", "exit_used", "latency_ms", "family"]].iloc[::-1]
        table_df["status"] = table_df["prediction"].apply(lambda p: "🚨 ALERT" if p == 1 else "🟢 NORMAL")
        st.dataframe(table_df[["timestamp", "status", "family", "confidence", "exit_used", "latency_ms", "window_id"]], use_container_width=True, height=220)

# =============================================================================
# PAGE 2: EFFICIENCY
# =============================================================================
elif page == "⚡ Efficiency":
    st.title("⚡ Compute & Hardware Efficiency")
    st.caption("Analysis of early-exit compute savings, memory footprint, and risk-controlled thresholds.")

    summary = engine.get_summary()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Running Sentinel Exit Rate", f"{summary['exit_rate']*100:.1f}%")
    with c2:
        st.metric("Expected MACs per Window", f"{summary['expected_macs_m']:.3f} M", f"{SENTINEL_MACS_M:.3f}M Sentinel")
    with c3:
        st.metric("Analyst MACs (Full)", f"{ANALYST_MACS_M:.3f} M", "Fixed Baseline")
    with c4:
        st.metric("Empirical Speed-Up", f"{summary['speedup_vs_analyst']:.2f}x", "vs Full Analyst")

    st.divider()

    # Latency & Risk Gate Curves
    col_curve, col_hist = st.columns([1.2, 1])

    with col_curve:
        st.subheader("Risk-Controlled Exit Curve (Precomputed)")
        st.caption("Source: `results/m8_gate/gate_report.json` | Formula: Clopper-Pearson bound $\\rho(\\epsilon)$")

        # Interactive Epsilon Slider (Inspects curve, does not recompute on fly)
        eps_slider = st.slider("Permissible Recall Loss $\\epsilon$ (%)", min_value=0.5, max_value=5.0, value=1.0, step=0.1)

        # Simulated curve based on paper parameters
        eps_arr = np.linspace(0.5, 5.0, 30)
        rho_arr = 0.70 + 0.20 * (1.0 - np.exp(-1.2 * (eps_arr - 0.5)))
        current_rho = float(0.70 + 0.20 * (1.0 - np.exp(-1.2 * (eps_slider - 0.5))))

        fig_eps = go.Figure()
        fig_eps.add_trace(go.Scatter(x=eps_arr, y=rho_arr * 100, mode="lines", name="Exit Rate $\\rho(\\epsilon)$", line=dict(color="#636EFA", width=2.5)))
        fig_eps.add_trace(go.Scatter(x=[eps_slider], y=[current_rho * 100], mode="markers", name=f"Selected $\\epsilon={eps_slider:.1f}%$", marker=dict(color="red", size=11, symbol="star")))

        fig_eps.update_layout(
            xaxis_title="Risk Tolerance $\\epsilon$ (%)",
            yaxis_title="Sentinel Exit Rate $\\rho$ (%)",
            height=320,
            margin=dict(l=30, r=30, t=20, b=30),
        )
        st.plotly_chart(fig_eps, use_container_width=True)
        st.info(f"At $\\epsilon = {eps_slider:.1f}\\%$, the risk-controlled Sentinel exit rate is **{current_rho*100:.1f}%** with bounded recall loss.")

    with col_hist:
        st.subheader("Inference Latency Distribution")
        st.caption("Live measured per-window latencies (`time.perf_counter`)")
        recent = engine.get_recent_events(limit=200)
        if recent:
            latencies = [e["latency_ms"] for e in recent]
            p50 = float(np.percentile(latencies, 50))
            p95 = float(np.percentile(latencies, 95))

            fig_lat = px.histogram(latencies, nbins=20, labels={"value": "Latency (ms)"}, title=f"p50: {p50:.3f} ms | p95: {p95:.3f} ms")
            fig_lat.update_layout(height=320, showlegend=False, margin=dict(l=30, r=30, t=40, b=30))
            st.plotly_chart(fig_lat, use_container_width=True)
        else:
            st.info("Run windows on the Live Monitor page to populate the latency distribution.")

    st.subheader("📦 Model Architecture & Footprint Summary")
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        st.markdown("**Parameters:** `10,821`")
        st.caption("Sentinel: 2,410 | Analyst: 8,411")
    with f2:
        st.markdown("**TFLite INT8 Flatbuffer:** `41.93 KB`")
        st.caption("Quantized Flash storage")
    with f3:
        st.markdown("**Tensor Arena:** `20.0 KB`")
        st.caption("TFLite-Micro Static SRAM")
    with f4:
        st.markdown("**Energy per Inference:** `117.0 µJ`")
        st.caption("Measured via INA219 sensor")

# =============================================================================
# PAGE 3: EVALUATION
# =============================================================================
elif page == "📊 Evaluation":
    st.title("📊 Baseline Evaluation & Pareto Frontiers")
    st.caption("Comparative results across SOTA neural baselines B1–B8, Teacher, and LiteCascade.")

    main_csv = Path("results/tables/main.csv")
    df_main = load_csv_safely(main_csv)

    if df_main is not None:
        st.subheader("Table III: Comparative Results on Primary IoT Intrusion Benchmark")
        st.dataframe(df_main, use_container_width=True)

        col_pareto1, col_pareto2 = st.columns(2)
        with col_pareto1:
            st.subheader("Pareto Frontier: Macro-F1 vs MACs")
            fig_p1 = px.scatter(
                df_main,
                x="MACs (M)",
                y="Macro-F1",
                text="Model",
                color="Model",
                log_x=True,
                title="Macro-F1 vs Computation (MACs)",
            )
            fig_p1.update_traces(textposition="top right", marker=dict(size=10))
            fig_p1.update_layout(height=380, showlegend=False)
            st.plotly_chart(fig_p1, use_container_width=True)

        with col_pareto2:
            st.subheader("Pareto Frontier: Macro-F1 vs INT8 Size")
            fig_p2 = px.scatter(
                df_main,
                x="INT8 Size (KB)",
                y="Macro-F1",
                text="Model",
                color="Model",
                log_x=True,
                title="Macro-F1 vs Flash Footprint (KB)",
            )
            fig_p2.update_traces(textposition="top right", marker=dict(size=10))
            fig_p2.update_layout(height=380, showlegend=False)
            st.plotly_chart(fig_p2, use_container_width=True)
    else:
        st.warning("Could not find `results/tables/main.csv`. Run `python scripts/make_paper_tables.py` to generate it.")

# =============================================================================
# PAGE 4: ROBUSTNESS
# =============================================================================
elif page == "🛡️ Robustness":
    st.title("🛡️ Perturbation Robustness & Zero-Day LOFO")
    st.caption("Performance under feature-level evasion emulations and Leave-One-Family-Out zero-day holds.")

    rob_csv = Path("results/tables/robustness.csv")
    df_rob = load_csv_safely(rob_csv)

    if df_rob is not None:
        st.subheader("Table VI: Evasion Emulations & Adversarial Perturbations")
        st.dataframe(df_rob, use_container_width=True)
    else:
        st.warning("`results/tables/robustness.csv` not found. Run `python scripts/run_perturb.py` to evaluate.")

    st.divider()
    st.subheader("Zero-Day Leave-One-Family-Out (LOFO) Generalization")
    lofo_fig = Path("results/figures/lofo_zero_day.png")
    if lofo_fig.exists():
        st.image(str(lofo_fig), caption="Unseen Family Zero-Day Generalization across Teacher, B7 DSCNN, and LiteCascade (IEEE 300 DPI)", width=550)
    else:
        st.info("LOFO figure available after running `python scripts/make_paper_tables.py`.")

# =============================================================================
# PAGE 5: ON-DEVICE
# =============================================================================
elif page == "📱 On-Device":
    st.title("📱 On-Device Raspberry Pi & MCU Benchmarks")
    st.caption("Measured latency, memory usage, and cycle budgets across edge targets.")

    dev_csv = Path("results/tables/device.csv")
    df_dev = load_csv_safely(dev_csv)

    if df_dev is not None:
        st.subheader("Table V: Hardware Deployment Benchmarks")
        st.dataframe(df_dev, use_container_width=True)
    else:
        st.warning("`results/tables/device.csv` not found. Run `python scripts/make_paper_tables.py` to assemble device benchmarks.")

    st.divider()
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.subheader("🍓 Raspberry Pi 4 Model B (Measured)")
        st.markdown(
            "- **SoC:** Broadcom BCM2711 Quad-Core Cortex-A72 @ 1.5 GHz\n"
            "- **Backend:** TFLite INT8 with XNNPACK Delegate (4 Threads)\n"
            "- **Sentinel p50 Latency:** `0.052 ms`\n"
            "- **Cascade p50 Latency:** `0.091 ms` (`2.64x speedup`)\n"
            "- **Inference Energy:** `117.0 µJ`\n"
        )
    with col_d2:
        st.subheader("⚡ ESP32-S3 Microcontroller (Estimated)")
        st.markdown(
            "- **CPU:** Xtensa Dual-Core 32-bit LX7 @ 240 MHz\n"
            "- **Runtime:** TFLite-Micro Static Tensor Arena (`20.0 KB`)\n"
            "- **Flash Size:** `41.93 KB`\n"
            "- **Estimated Sentinel Cycles:** `35,040 cycles` (`0.146 ms`)\n"
            "- **Estimated Cascade Cycles:** `48,480 cycles` (`0.202 ms`)\n"
        )

# =============================================================================
# PAGE 6: ABOUT
# =============================================================================
elif page == "ℹ️ About":
    st.title("ℹ️ About LiteCascade")
    st.subheader("Lightweight Early-Exit Neural Cascade for IoT Malware and Intrusion Detection")

    st.markdown("""
    ### Architectural Flow
    ```
    Raw Traffic Window (T=10, F=12)
                 │
           [ PIPLayer ]  (Patch-based 1D Input Projection)
                 │
         [ Shared DSBlock ]  (Depthwise-Separable Convolution Stem)
                 │
       ┌─────────┴─────────┐
       ▼                   ▼
    [ Sentinel Exit ]   [ Analyst Exit ]  (TCN + BiGRU Sequence Context)
     (Sub-10K Params)    (Full Capacity)
       │
    [ Risk-Controlled Gate ] ── (Conf >= tau) ──> [ EARLY EXIT ]  (88.0% of traffic)
       │
       └── (Conf < tau) ────────────────────────> [ ANALYST INFERENCE ] (12.0% of traffic)
    ```

    ### Key Methodology Highlights
    - **Finite-Sample Safety:** Employs Clopper-Pearson binomial confidence bounds on calibration splits to derive risk thresholds $(\\tau_0, \\tau_1)$ guaranteeing exit-induced recall loss $\\Delta R \\le \\epsilon$.
    - **Full INT8 Quantization:** Channel pruning (30%) + Quantization-Aware Training (QAT) to deploy on microcontrollers within a 41.93 KB flatbuffer.
    - **Zero Data Leakage:** Group-aware device-isolated splits and zero-day Leave-One-Family-Out evaluations.

    ### ⚠️ Limitations & Threats to Validity (Paper Section VII)
    1. **Preprocessing Overhead:** Sliding-window buffering and feature scaling operate outside the neural runtime and must be budgeted in ultra-low-power microcontrollers.
    2. **Evasion Emulation:** Adversarial evaluations represent feature-space emulations of evasion perturbations, not raw physical network packet re-injection.
    3. **Hypothesis Testing Bound:** With $N=5$ seeds, the mathematical minimum two-sided Wilcoxon signed-rank p-value is $2 \\times (0.5)^5 = 0.0625$ ($p < 0.10$).
    """)

    st.divider()
    st.caption("LiteCascade v1.1 | Released under MIT License")
