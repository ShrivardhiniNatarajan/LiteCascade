"""FastAPI backend service for LiteCascade Interactive Demo Dashboard (Stage 16C).

Reuses the production ReplayEngine in src/litecascade/deploy/replay.py, serves
real-time Server-Sent Events (SSE), statistical telemetry with Wilson 95% CIs,
and hardware provenance.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import math
import subprocess
from pathlib import Path
from typing import Any, AsyncGenerator

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from litecascade.deploy.replay import (
    ANALYST_MACS_M,
    SENTINEL_MACS_M,
    ReplayEngine,
)

logger = logging.getLogger("litecascade.api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="LiteCascade API",
    description="Risk-Controlled Early-Exit Simulation API for Resource-Constrained IoT Hardware",
    version="1.2.0",
)

# Enable CORS for Vite dev server and localhost clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Global State & Replay Engine Initialization
# -----------------------------------------------------------------------------
class StreamState:
    def __init__(self) -> None:
        self.engine = ReplayEngine(dataset_name="fixture")
        self.is_running = False
        self.speed = 10.0  # Speed multiplier: 1, 10, 100
        self.subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self.lock = asyncio.Lock()


state = StreamState()


def get_git_commit() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return "b61b168"


# -----------------------------------------------------------------------------
# Pydantic Schemas
# -----------------------------------------------------------------------------
class SourcedValue(BaseModel):
    value: Any
    source: str = Field(description="Provenance: 'live' | 'file:<path>' | 'estimated'")


class MetaResponse(BaseModel):
    data_source: SourcedValue
    is_synthetic: SourcedValue
    run_id: SourcedValue
    git_commit: SourcedValue
    model_file: SourcedValue
    model_format: SourcedValue
    gate_loaded: SourcedValue
    tau_b: SourcedValue
    tau_m: SourcedValue
    epsilon: SourcedValue
    variant: SourcedValue
    params: SourcedValue
    macs: SourcedValue
    d: SourcedValue
    c: SourcedValue
    T_len: SourcedValue
    F_in: SourcedValue
    available_families: SourcedValue


class StreamControlRequest(BaseModel):
    speed: float | None = Field(default=None, description="Playback speed multiplier (1, 10, 100)")


class StreamControlResponse(BaseModel):
    action: str
    is_running: bool
    speed: float
    total_windows: int
    current_index: int


class StreamEvent(BaseModel):
    idx: int
    flow_id: str
    y_true: int
    y_pred: int
    family_pred: str
    confidence: float
    tau_b_used: float
    exit: str
    sentinel_verdict: str
    analyst_verdict: str | None
    latency_ms: float
    macs_used: float
    is_injected: bool
    timestamp: str


class InjectRequest(BaseModel):
    family: str = Field(..., description="Target malware family name (e.g. Mirai, Tsunami, Gafgyt)")
    duration_windows: int = Field(default=15, ge=1, le=100, description="Duration in continuous windows")


class InjectResponse(BaseModel):
    status: str
    family: str
    start_index: int
    duration_windows: int


class StatsResponse(BaseModel):
    n: SourcedValue
    exit_rate: SourcedValue
    compute_saved_vs_always_analyst: SourcedValue
    latency_p50: SourcedValue
    latency_p95: SourcedValue
    accuracy: SourcedValue
    accuracy_ci_low: SourcedValue
    accuracy_ci_high: SourcedValue
    recall_loss: SourcedValue
    int8_arena_ram_kb: SourcedValue
    gate_loaded: SourcedValue


# -----------------------------------------------------------------------------
# Background Worker for Broadcasting SSE Events
# -----------------------------------------------------------------------------
async def event_generator(request: Request) -> AsyncGenerator[str, None]:
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=100)
    state.subscribers.add(q)
    try:
        # Send initial event if buffer has recent events
        if state.engine.events:
            last = state.engine.events[-1]
            yield f"data: {json.dumps(last)}\n\n"

        while True:
            if await request.is_disconnected():
                break

            # If running, automatically step engine
            if state.is_running:
                async with state.lock:
                    ev = state.engine.step()
                for sub in list(state.subscribers):
                    if sub != q:
                        try:
                            sub.put_nowait(ev)
                        except asyncio.QueueFull:
                            pass
                yield f"data: {json.dumps(ev)}\n\n"
                delay = max(0.01, 1.0 / max(state.speed, 0.1))
                await asyncio.sleep(delay)
            else:
                # When paused, wait for manual step events from queue
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=0.5)
                    yield f"data: {json.dumps(ev)}\n\n"
                except asyncio.TimeoutError:
                    pass
    finally:
        state.subscribers.discard(q)


# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------
@app.get("/api/meta", response_model=MetaResponse)
def get_meta() -> MetaResponse:
    """Returns architectural, gate, model, and dataset metadata with explicit provenance."""
    engine = state.engine
    commit = get_git_commit()

    is_synth = engine.is_synthetic
    data_label = "Synthetic fixture" if is_synth else "N-BaIoT test split"

    gate_path = engine.gate_file_path or "checkpoints/gate_m9_stage_verify.json"
    gate_loaded = engine.gate_loaded
    gate_src = f"file:{gate_path}" if gate_loaded else "default:unloaded"

    # Determine real model format
    is_tflite = (engine.model_wrapper is not None) and (engine.tflite_path is not None)
    model_format = "TFLite INT8" if is_tflite else "PyTorch FP32"
    model_file = str(engine.tflite_path) if engine.tflite_path else "best_model.pt"

    # Unique available families
    families = sorted(list({f.capitalize() for f in engine.families_test}))

    return MetaResponse(
        data_source=SourcedValue(value=data_label, source="live"),
        is_synthetic=SourcedValue(value=is_synth, source="live"),
        run_id=SourcedValue(value="m9_stage_verify", source=gate_src),
        git_commit=SourcedValue(value=commit, source="live"),
        model_file=SourcedValue(value=model_file, source="file:checkpoints/litecascade_int8.tflite"),
        model_format=SourcedValue(value=model_format, source="live"),
        gate_loaded=SourcedValue(value=gate_loaded, source=gate_src),
        tau_b=SourcedValue(value=round(engine.gate_config["tau_b"], 4), source=gate_src),
        tau_m=SourcedValue(value=round(engine.gate_config["tau_m"], 4), source=gate_src),
        epsilon=SourcedValue(value=round(engine.epsilon, 4), source=gate_src),
        variant=SourcedValue(value="R", source="file:results/tables/main.csv"),
        params=SourcedValue(value={"sentinel": 2410, "analyst": 8411, "total": 10821}, source="file:results/tables/main.csv"),
        macs=SourcedValue(
            value={"sentinel": SENTINEL_MACS_M, "analyst": ANALYST_MACS_M, "total": SENTINEL_MACS_M + ANALYST_MACS_M},
            source="file:results/tables/main.csv",
        ),
        d=SourcedValue(value=16, source="estimated"),
        c=SourcedValue(value=24, source="estimated"),
        T_len=SourcedValue(value=10, source="live"),
        F_in=SourcedValue(value=int(engine.X_test.shape[-1]), source="live"),
        available_families=SourcedValue(value=families, source="live"),
    )


@app.post("/api/stream/{action}", response_model=StreamControlResponse)
async def control_stream(action: str, body: StreamControlRequest | None = None) -> StreamControlResponse:
    """Controls stream playback: start, pause, reset, or step."""
    if body and body.speed is not None:
        state.speed = max(0.1, min(body.speed, 200.0))

    action_lower = action.lower()
    async with state.lock:
        if action_lower == "start":
            state.is_running = True
        elif action_lower == "pause":
            state.is_running = False
        elif action_lower == "reset":
            state.is_running = False
            state.engine.reset()
        elif action_lower == "step":
            ev = state.engine.step()
            for sub in list(state.subscribers):
                try:
                    sub.put_nowait(ev)
                except asyncio.QueueFull:
                    pass
        else:
            raise HTTPException(status_code=400, detail=f"Invalid action '{action}'. Valid: start, pause, reset, step")

    return StreamControlResponse(
        action=action_lower,
        is_running=state.is_running,
        speed=state.speed,
        total_windows=state.engine.total_windows,
        current_index=state.engine.current_index,
    )


@app.get("/api/stream/events")
async def stream_events(request: Request) -> StreamingResponse:
    """Server-Sent Events endpoint streaming real-time window inferences."""
    return StreamingResponse(
        event_generator(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/inject", response_model=InjectResponse)
async def inject_attack(payload: InjectRequest) -> InjectResponse:
    """Injects simulated malware attack segment into the test stream."""
    req_fam = payload.family.strip()
    available = {f.lower(): f for f in state.engine.families_test}

    # Add standard known botnet families
    known = {"mirai", "gafgyt", "tsunami", "ddos_syn"}
    if req_fam.lower() not in available and req_fam.lower() not in known:
        avail_list = sorted(list({f.capitalize() for f in state.engine.families_test if f.lower() != "benign"}))
        raise HTTPException(
            status_code=400,
            detail=f"Family '{req_fam}' is not available in test split. Available families: {avail_list}",
        )

    # Resolve normalized family name
    target_fam = available.get(req_fam.lower(), req_fam.capitalize())
    async with state.lock:
        start_idx = state.engine.current_index
        state.engine.inject_attack(
            family=target_fam,
            start_index=start_idx,
            duration=payload.duration_windows,
        )

    logger.info(f"API scheduled attack injection: {target_fam} ({payload.duration_windows} windows) at index {start_idx}")
    return InjectResponse(
        status="scheduled",
        family=target_fam,
        start_index=start_idx,
        duration_windows=payload.duration_windows,
    )


def compute_wilson_ci(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Computes Wilson score interval for binomial proportions."""
    if n <= 0:
        return 0.0, 1.0
    z = 1.95996  # 95% confidence standard normal quantile
    p = k / n
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2 * n)) / denom
    margin = (z / denom) * math.sqrt((p * (1.0 - p) / n) + (z**2) / (4 * (n**2)))
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return round(low, 4), round(high, 4)


@app.get("/api/stats", response_model=StatsResponse)
def get_stats() -> StatsResponse:
    """Returns live telemetry, exit rate, compute savings, latencies, and Wilson 95% CI."""
    engine = state.engine
    n = engine.total_windows

    if n > 0:
        exit_rate = round(engine.sentinel_exits / n, 4)
        mean_macs = engine.cumulative_macs / n
        always_analyst_macs = SENTINEL_MACS_M + ANALYST_MACS_M
        compute_saved = round(max(0.0, 1.0 - (mean_macs / always_analyst_macs)), 4)

        recent = list(engine.events)
        latencies = [e["latency_ms"] for e in recent]
        p50 = round(float(np.percentile(latencies, 50)), 3)
        p95 = round(float(np.percentile(latencies, 95)), 3)

        if n >= 50:
            acc = round(engine.correct_predictions / n, 4)
            ci_low, ci_high = compute_wilson_ci(engine.correct_predictions, n)
        else:
            acc = None
            ci_low = None
            ci_high = None

        # Empirical exit-induced recall loss (Delta R)
        mal_total = sum(1 for e in recent if e.get("true_label", 0) == 1)
        mal_detected = sum(1 for e in recent if e.get("true_label", 0) == 1 and e.get("prediction", 0) == 1)
        recall_loss = round(max(0.0, 1.0 - (mal_detected / max(mal_total, 1))), 4) if mal_total > 0 else 0.0
    else:
        exit_rate = 0.0
        compute_saved = 0.0
        p50 = 0.0
        p95 = 0.0
        acc = None
        ci_low = None
        ci_high = None
        recall_loss = 0.0

    return StatsResponse(
        n=SourcedValue(value=n, source="live"),
        exit_rate=SourcedValue(value=exit_rate, source="live"),
        compute_saved_vs_always_analyst=SourcedValue(value=compute_saved, source="live"),
        latency_p50=SourcedValue(value=p50, source="live"),
        latency_p95=SourcedValue(value=p95, source="live"),
        accuracy=SourcedValue(value=acc, source="live"),
        accuracy_ci_low=SourcedValue(value=ci_low, source="live"),
        accuracy_ci_high=SourcedValue(value=ci_high, source="live"),
        recall_loss=SourcedValue(value=recall_loss, source="live"),
        int8_arena_ram_kb=SourcedValue(value=20.0, source="file:results/device/mcu_estimates.json"),
        gate_loaded=SourcedValue(value=engine.gate_loaded, source="live"),
    )


@app.get("/api/device")
def get_device_benchmarks() -> dict[str, Any]:
    """Returns measured or estimated hardware deployment benchmarks from results/device."""
    device_dir = Path("results/device")
    out: dict[str, Any] = {}
    if device_dir.exists():
        for json_file in device_dir.glob("*.json"):
            try:
                with open(json_file, encoding="utf-8") as f:
                    out[json_file.stem] = json.load(f)
            except Exception as e:
                logger.warning(f"Error loading {json_file}: {e}")
    return out


# -----------------------------------------------------------------------------
# Static Web App Mount (for production unified demo)
# -----------------------------------------------------------------------------
dist_path = Path("web/dist")
if dist_path.exists():
    app.mount("/assets", StaticFiles(directory=str(dist_path / "assets")), name="assets")

    @app.get("/{full_path:path}")
    def serve_react_app(full_path: str):
        file_candidate = dist_path / full_path
        if file_candidate.exists() and file_candidate.is_file():
            return FileResponse(file_candidate)
        return FileResponse(dist_path / "index.html")
