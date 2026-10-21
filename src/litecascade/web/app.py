"""LiteCascade dashboard backend (FastAPI). Serves the React dashboard and a JSON / SSE API.

    python -m litecascade.web.app                 # http://127.0.0.1:8000
    python -m litecascade.web.app --port 8080 --artifacts artifacts

API
    GET  /api/models                 model zoo cards (LiteCascade + deep-learning baselines) with stored test metrics
    GET  /api/architecture           LiteCascade structure, per-path parameters / MACs, calibrated gate
    GET  /api/live/state             replay state + running per-model metrics
    POST /api/live/control           {"action": "play"|"pause"|"step"|"reset", "speed": 1..100}
    POST /api/live/inject            {"family": "...", "windows": 30}  queue a burst of real held-out attack windows
    POST /api/live/models            {"active": [model ids]}            choose the models raced in the arena
    GET  /api/live/stream            Server-Sent Events: one event per replayed window
    GET  /api/gate/curve             rho / miss-rate / F1 vs epsilon from the cascade's calibration outputs
    GET  /api/gate/point             same for one (epsilon, alpha)
    POST /api/analyze                multipart file=<csv>, models=<comma separated ids>
    POST /api/analyze_sample         form sample=<examples csv>, models=<ids>
    GET  /api/download/{job_id}      predictions CSV (all selected models)
    GET  /api/samples[/{name}]       example CSVs (synthetic, unseen test devices)
    GET  /api/docs[/{name}]          project documentation and evaluation reports (Markdown)
"""

import argparse
import asyncio
import io
import json
import os
import threading
import uuid
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from litecascade.deploy.predict import predict_dataframe
from litecascade.web.live import GateLab, ModelZoo, ReplayEngine

DIST = Path(__file__).parent / "dashboard_dist"
ARTIFACTS = Path(os.environ.get("LITECASCADE_ARTIFACTS", "artifacts"))
EXAMPLES = Path(os.environ.get("LITECASCADE_EXAMPLES", "examples"))
DOC_DIRS = [Path("docs"), Path(os.environ.get("LITECASCADE_REPORTS", "reports"))]
MAX_UPLOAD_MB = 200
MAX_ROWS_RETURNED = 3000

app = FastAPI(title="LiteCascade dashboard API", version="2.0")


class _State:
    def __init__(self):
        self.lock = threading.Lock()
        self.zoo: ModelZoo | None = None
        self.engine: ReplayEngine | None = None
        self.gate_labs: dict[str, GateLab] = {}
        self.running = False
        self.speed = 10.0
        self.subscribers: set[asyncio.Queue] = set()
        self.producer: asyncio.Task | None = None
        self.jobs: OrderedDict[str, bytes] = OrderedDict()

    def get_zoo(self) -> ModelZoo:
        with self.lock:
            if self.zoo is None:
                try:
                    self.zoo = ModelZoo(ARTIFACTS)
                except FileNotFoundError as e:
                    raise HTTPException(503, str(e)) from e
            return self.zoo

    def get_engine(self) -> ReplayEngine:
        zoo = self.get_zoo()
        with self.lock:
            if self.engine is None:
                self.engine = ReplayEngine(zoo)
            return self.engine


S = _State()


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.integer | np.bool_):
        return o.item()
    if isinstance(o, np.floating):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    return o


# ---------------------------------------------------------------------------------------------- zoo
@app.get("/api/health")
def health():
    return {"ok": True, "dashboard_built": (DIST / "index.html").exists()}


@app.get("/api/models")
def models():
    return _jsonable(S.get_zoo().cards())


@app.get("/api/architecture")
def architecture():
    zoo = S.get_zoo()
    b = zoo.primary()
    if b.kind != "cascade":
        raise HTTPException(404, "No LiteCascade bundle loaded")
    card = zoo.card(b)
    return _jsonable(
        {
            "model": card,
            "config": b.meta["model_config"],
            "gate": b.gate,
            "params": card["metrics"].get("param_breakdown"),
            "macs": {
                "sentinel_path": card["metrics"].get("macs_sentinel"),
                "full": card["metrics"].get("macs_full"),
                "expected": card["metrics"].get("macs"),
            },
            "feature_preview": b.feature_cols[:8],
        }
    )


# ---------------------------------------------------------------------------------------------- live replay
def _live_payload(event=None):
    eng = S.get_engine()
    return _jsonable({"event": event, "summary": eng.summary(), "running": S.running, "speed": S.speed})


async def _producer():
    eng = S.get_engine()
    loop = asyncio.get_running_loop()
    while S.running:
        event = await loop.run_in_executor(None, eng.step)
        payload = json.dumps(_live_payload(event))
        for q in list(S.subscribers):
            if q.qsize() < 200:
                q.put_nowait(payload)
        await asyncio.sleep(max(0.0, 1.0 / max(S.speed, 0.1) * 0.5))


@app.get("/api/live/state")
def live_state():
    eng = S.get_engine()
    return _jsonable(
        {
            **_live_payload(),
            "families": eng.families,
            "active": eng.active,
            "primary": eng.primary.name,
            "test_windows": int(len(eng.y)),
            "timeline": list(eng.timeline),
        }
    )


@app.post("/api/live/control")
async def live_control(body: dict = Body(...)):
    action = body.get("action")
    if "speed" in body:
        S.speed = float(np.clip(float(body["speed"]), 0.5, 100))
    eng = S.get_engine()
    if action == "play":
        S.running = True
        if S.producer is None or S.producer.done():
            S.producer = asyncio.create_task(_producer())
    elif action == "pause":
        S.running = False
    elif action == "reset":
        eng.reset()
    elif action == "step":
        event = await asyncio.get_running_loop().run_in_executor(None, eng.step)
        payload = json.dumps(_live_payload(event))
        for q in list(S.subscribers):
            q.put_nowait(payload)
        return json.loads(payload)
    elif action is not None:
        raise HTTPException(400, f"unknown action {action}")
    return _live_payload()


@app.post("/api/live/inject")
def live_inject(body: dict = Body(...)):
    try:
        return S.get_engine().inject(body.get("family", ""), int(body.get("windows", 30)))
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@app.post("/api/live/models")
def live_models(body: dict = Body(...)):
    eng = S.get_engine()
    active = [m for m in body.get("active", []) if m in eng.zoo.bundles]
    if eng.primary.name not in active:
        active.insert(0, eng.primary.name)
    eng.active = active
    return {"active": active}


@app.get("/api/live/stream")
async def live_stream():
    q: asyncio.Queue = asyncio.Queue()
    S.subscribers.add(q)

    async def gen():
        try:
            yield f"data: {json.dumps(_live_payload())}\n\n"
            while True:
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=15)
                    yield f"data: {msg}\n\n"
                except TimeoutError:
                    yield ": keep-alive\n\n"
        finally:
            S.subscribers.discard(q)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


# ---------------------------------------------------------------------------------------------- gate lab
def _gate_lab(model: str | None) -> GateLab:
    zoo = S.get_zoo()
    b = zoo.bundles.get(model) if model else zoo.primary()
    if b is None or b.kind != "cascade":
        raise HTTPException(404, "Gate lab needs a LiteCascade bundle")
    if b.name not in S.gate_labs:
        S.gate_labs[b.name] = GateLab(b, S.get_engine().data)
    return S.gate_labs[b.name]


@app.get("/api/gate/curve")
def gate_curve(model: str | None = None, alpha: float = 0.05):
    lab = _gate_lab(model)
    return _jsonable({"model": lab.b.name, "deployed": lab.b.gate, "curve": lab.curve(alpha)})


@app.get("/api/gate/point")
def gate_point(epsilon: float, alpha: float = 0.05, model: str | None = None):
    if not (0 < epsilon < 1 and 0 < alpha < 1):
        raise HTTPException(400, "epsilon and alpha must be in (0, 1)")
    return _jsonable(_gate_lab(model).point(epsilon, alpha))


# ---------------------------------------------------------------------------------------------- CSV analysis
def _analyze(df: pd.DataFrame, model_ids: list[str]) -> dict:
    zoo = S.get_zoo()
    ids = [m for m in model_ids if m in zoo.bundles] or [zoo.primary().name]
    results, combined = {}, None
    for mid in ids:
        try:
            res, summary = predict_dataframe(zoo.bundles[mid], df)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        results[mid] = {"summary": summary, "res": res}
        cols = res[["row", "segment", "prediction", "malicious_score"] + (["exit_stage"] if mid == ids[0] else [])]
        cols = cols.rename(columns={c: f"{mid}:{c}" for c in cols.columns if c not in ("row", "segment")})
        if combined is None:
            combined = cols.copy()
            if "true_label" in res.columns:
                combined.insert(2, "true_label", res["true_label"].to_numpy())
        else:
            combined = combined.merge(cols, on=["row", "segment"], how="left")
    job = uuid.uuid4().hex[:12]
    S.jobs[job] = combined.to_csv(index=False).encode()
    while len(S.jobs) > 20:
        S.jobs.popitem(last=False)
    first = results[ids[0]]["res"]
    return _jsonable(
        {
            "job_id": job,
            "models": ids,
            "summaries": {k: v["summary"] for k, v in results.items()},
            "rows": first.head(MAX_ROWS_RETURNED).replace({np.nan: None}).to_dict(orient="records"),
            "rows_truncated": len(first) > MAX_ROWS_RETURNED,
            "scores": {k: v["res"]["malicious_score"].round(4).head(MAX_ROWS_RETURNED).tolist() for k, v in results.items()},
        }
    )


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), models: str = Form("")):
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File larger than {MAX_UPLOAD_MB} MB")
    try:
        df = pd.read_csv(io.BytesIO(raw), low_memory=False)
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}") from e
    return await asyncio.get_running_loop().run_in_executor(None, _analyze, df, [m for m in models.split(",") if m])


def _sample_path(name: str) -> Path:
    for p in sorted(EXAMPLES.glob("*.csv")):
        if p.name == name:
            return p
    raise HTTPException(404, f"Unknown sample '{name}'")


@app.post("/api/analyze_sample")
async def analyze_sample(sample: str = Form(...), models: str = Form("")):
    df = pd.read_csv(_sample_path(sample))
    return await asyncio.get_running_loop().run_in_executor(None, _analyze, df, [m for m in models.split(",") if m])


@app.get("/api/download/{job_id}")
def download(job_id: str):
    if job_id not in S.jobs:
        raise HTTPException(404, "Result expired; run the analysis again.")
    return Response(
        S.jobs[job_id], media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="litecascade_{job_id}.csv"'}
    )


@app.get("/api/samples")
def samples():
    return [{"name": p.name, "kb": round(p.stat().st_size / 1024, 1)} for p in sorted(EXAMPLES.glob("*.csv"))]


@app.get("/api/samples/{name}")
def sample_file(name: str):
    return FileResponse(_sample_path(name), media_type="text/csv", filename=name)


# ---------------------------------------------------------------------------------------------- docs
def _docs() -> dict[str, Path]:
    out = {}
    for d in DOC_DIRS:
        for p in sorted(d.glob("*.md")):
            out[p.name] = p
    return out


@app.get("/api/docs")
def docs():
    return [
        {"name": n, "title": p.read_text(encoding="utf-8").split("\n", 1)[0].lstrip("# ").strip()} for n, p in _docs().items()
    ]


@app.get("/api/docs/{name}", response_class=PlainTextResponse)
def doc(name: str):
    p = _docs().get(name)
    if p is None:
        raise HTTPException(404, "Unknown document")
    return p.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------------------------- frontend
if (DIST / "assets").exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")


@app.get("/{path:path}", response_class=HTMLResponse, include_in_schema=False)
def spa(path: str):
    if path.startswith("api/"):
        raise HTTPException(404, "Unknown API route")
    candidate = (DIST / path).resolve()
    if path and candidate.is_file() and DIST.resolve() in candidate.parents:
        return FileResponse(candidate)
    index = DIST / "index.html"
    if not index.exists():
        return HTMLResponse(
            "<h2>Dashboard not built</h2><p>Run <code>cd dashboard && npm ci && npm run build</code>. "
            "The API is live under /api.</p>"
        )
    return HTMLResponse(index.read_text(encoding="utf-8"))


def main():
    global ARTIFACTS, EXAMPLES
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--artifacts", default=str(ARTIFACTS))
    ap.add_argument("--examples", default=str(EXAMPLES))
    args = ap.parse_args()
    ARTIFACTS, EXAMPLES = Path(args.artifacts), Path(args.examples)
    import uvicorn

    print(f"LiteCascade dashboard on http://{args.host}:{args.port}  (models from {ARTIFACTS.resolve()})")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
