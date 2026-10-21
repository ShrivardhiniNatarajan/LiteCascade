import { motion } from "framer-motion";
import { Download, FileUp } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Bar } from "../components/ui";
import { api, fmt } from "../lib/api";
import { useLive } from "../lib/live";

type Summary = {
  display_name: string;
  model_kind: string;
  windows: number;
  records: number;
  predicted_malicious: number;
  exit_rate: number;
  analyst_invocations: number;
  inference_ms_per_window: number;
  metrics?: { f1_macro: number; recall_malicious: number; fpr: number; accuracy: number; tp: number; fp: number; fn: number; tn: number };
};
type Result = {
  job_id: string;
  models: string[];
  summaries: Record<string, Summary>;
  rows: Record<string, string | number | boolean | null>[];
  rows_truncated: boolean;
  scores: Record<string, number[]>;
};

function Overlay({ scores, labels }: { scores: Record<string, number[]>; labels: (string | null)[] }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const w = c.clientWidth, h = 200, dpr = window.devicePixelRatio || 1;
    c.width = w * dpr;
    c.height = h * dpr;
    const ctx = c.getContext("2d")!;
    ctx.scale(dpr, dpr);
    const css = getComputedStyle(document.documentElement);
    const col = (n: string) => css.getPropertyValue(n).trim();
    const n = labels.length;
    ctx.fillStyle = col("--malicious") + "22";
    labels.forEach((l, i) => {
      if (l === "malicious") ctx.fillRect((i / n) * w, 0, w / n + 1, h);
    });
    const palette = [col("--accent"), col("--analyst"), col("--ok"), col("--warn")];
    Object.values(scores).forEach((s, k) => {
      ctx.strokeStyle = palette[k % palette.length];
      ctx.lineWidth = 1.3;
      ctx.globalAlpha = 0.9;
      ctx.beginPath();
      s.forEach((v, i) => {
        const x = (i / n) * w, y = 6 + (1 - v) * (h - 12);
        if (i) ctx.lineTo(x, y);
        else ctx.moveTo(x, y);
      });
      ctx.stroke();
    });
    ctx.globalAlpha = 1;
  }, [scores, labels]);
  return <canvas ref={ref} style={{ width: "100%", height: 200, display: "block" }} />;
}

export function AnalyzePage() {
  const { cards, primary } = useLive();
  const [samples, setSamples] = useState<{ name: string; kb: number }[]>([]);
  const [sel, setSel] = useState<string[]>([]);
  const [file, setFile] = useState<File | null>(null);
  const [over, setOver] = useState(false);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [res, setRes] = useState<Result | null>(null);
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => {
    api.get<{ name: string; kb: number }[]>("/api/samples").then(setSamples).catch(() => undefined);
  }, []);
  useEffect(() => {
    if (primary && sel.length === 0) setSel([primary, "fixture_teacher", "fixture_b10"].filter((m) => cards.some((c) => c.id === m)));
  }, [primary, cards, sel.length]);

  const run = async (sample?: string) => {
    setBusy(true);
    setErr(null);
    try {
      const fd = new FormData();
      fd.append("models", sel.join(","));
      if (sample) {
        fd.append("sample", sample);
        setRes(await api.form<Result>("/api/analyze_sample", fd));
      } else if (file) {
        fd.append("file", file);
        setRes(await api.form<Result>("/api/analyze", fd));
      }
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  const labels = res ? res.rows.map((r) => (r.true_label as string) ?? null) : [];

  return (
    <>
      <div className="page-head">
        <div>
          <h2>Analyze traffic</h2>
          <p>Upload a CSV of flow records (the model's feature columns, optional segment / timestamp / label) and compare models on it.</p>
        </div>
      </div>
      <div className="grid g2">
        <div className="panel">
          <div className="panel-head"><h3>1 · Data</h3></div>
          <div className={`drop ${over ? "over" : ""}`} onClick={() => input.current?.click()} tabIndex={0}
            onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)}
            onDrop={(e) => { e.preventDefault(); setOver(false); setFile(e.dataTransfer.files[0] ?? null); }}>
            <FileUp size={22} />
            <div>{file ? `${file.name} (${(file.size / 1024).toFixed(0)} KB)` : "Drop a CSV here or click to browse"}</div>
            <input ref={input} type="file" accept=".csv" hidden onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
          </div>
          <div className="toolbar" style={{ marginTop: 12 }}>
            <button className="btn primary" disabled={!file || busy || !sel.length} onClick={() => run()}>{busy ? "Running…" : "Analyze file"}</button>
            {samples.map((s) => (
              <button key={s.name} className="btn" disabled={busy || !sel.length} onClick={() => run(s.name)}>Sample: {s.name}</button>
            ))}
          </div>
          <p className="hint">Samples are raw records of the two held-out synthetic test devices.</p>
        </div>
        <div className="panel">
          <div className="panel-head"><h3>2 · Models</h3><span className="hint">{sel.length} selected</span></div>
          <div className="toolbar">
            {cards.map((c) => (
              <button key={c.id} className={`chip ${sel.includes(c.id) ? "on" : ""}`}
                onClick={() => setSel(sel.includes(c.id) ? sel.filter((x) => x !== c.id) : [...sel, c.id])}>{c.name}</button>
            ))}
          </div>
        </div>
      </div>
      {err && <div className="status" style={{ marginTop: 16, borderColor: "var(--malicious)" }}>{err}</div>}
      {res && (
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} style={{ marginTop: 16 }}>
          <div className="panel" style={{ marginBottom: 16 }}>
            <div className="panel-head">
              <h3>Per-model results</h3>
              <a className="btn" href={`/api/download/${res.job_id}`}><Download size={15} /> Download CSV (all models)</a>
            </div>
            <div className="scroll">
              <table className="t">
                <thead><tr><th>Model</th><th>Windows</th><th>Flagged</th><th>Sentinel exits</th><th>ms / window</th><th>macro-F1</th><th>Recall</th><th>FPR</th><th>Errors</th></tr></thead>
                <tbody>
                  {res.models.map((m) => {
                    const s = res.summaries[m];
                    return (
                      <tr key={m}>
                        <td><b>{s.display_name}</b></td>
                        <td>{fmt.int(s.windows)}</td>
                        <td>{fmt.pct(s.predicted_malicious / s.windows)}</td>
                        <td>{s.model_kind === "cascade" ? fmt.pct(s.exit_rate) : "n/a"}</td>
                        <td>{s.inference_ms_per_window.toFixed(3)}</td>
                        <td>{s.metrics ? <div style={{ minWidth: 110 }}>{fmt.num(s.metrics.f1_macro)}<Bar value={s.metrics.f1_macro} max={1} height={5} /></div> : "–"}</td>
                        <td>{fmt.num(s.metrics?.recall_malicious)}</td>
                        <td>{fmt.num(s.metrics?.fpr, 4)}</td>
                        <td>{s.metrics ? `${s.metrics.fp} FP / ${s.metrics.fn} FN` : "–"}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
          <div className="panel">
            <div className="panel-head">
              <h3>p(malicious) per window, all selected models</h3>
              <div className="toolbar hint">
                {res.models.map((m, k) => (
                  <span key={m} style={{ color: ["var(--accent)", "var(--analyst)", "var(--ok)", "var(--warn)"][k % 4] }}>━ {res.summaries[m].display_name}</span>
                ))}
                {labels[0] != null && <span style={{ color: "var(--malicious)" }}>▮ labelled malicious</span>}
              </div>
            </div>
            <Overlay scores={res.scores} labels={labels} />
          </div>
        </motion.div>
      )}
    </>
  );
}
