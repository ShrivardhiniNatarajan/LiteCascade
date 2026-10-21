import { motion } from "framer-motion";
import { useMemo, useState } from "react";
import { Bar, useElementWidth, useTooltip } from "../components/ui";
import { fmt, type ModelCard } from "../lib/api";
import { useLive } from "../lib/live";

type XKey = "macs" | "params" | "size_kb" | "latency_p50_ms";
const XOPTS: { id: XKey; label: string }[] = [
  { id: "macs", label: "MACs / window" },
  { id: "params", label: "Parameters" },
  { id: "size_kb", label: "Model size (KB)" },
  { id: "latency_p50_ms", label: "Latency p50 (ms)" },
];

function Pareto({ cards, xKey }: { cards: ModelCard[]; xKey: XKey }) {
  const [ref, width] = useElementWidth<HTMLDivElement>(700);
  const tip = useTooltip();
  const H = 360, L = 56, R = 20, T = 16, B = 44;
  const pts = cards.filter((c) => c.metrics[xKey] != null && c.metrics.f1_macro != null);
  const xs = pts.map((c) => Math.log10(c.metrics[xKey] as number));
  const ys = pts.map((c) => c.metrics.f1_macro as number);
  const xmin = Math.floor(Math.min(...xs, 0) * 2) / 2, xmax = Math.ceil(Math.max(...xs, 1) * 2) / 2;
  const ymin = Math.floor((Math.min(...ys, 1) - 0.01) * 50) / 50, ymax = Math.min(1, Math.ceil((Math.max(...ys, 0) + 0.005) * 50) / 50);
  const X = (v: number) => L + ((Math.log10(v) - xmin) / (xmax - xmin || 1)) * (width - L - R);
  const Y = (v: number) => T + (1 - (v - ymin) / (ymax - ymin || 1)) * (H - T - B);
  const pmax = Math.max(...pts.map((c) => c.metrics.params ?? 1));
  // Pareto front: lower x, higher F1
  const sorted = [...pts].sort((a, b) => (a.metrics[xKey] as number) - (b.metrics[xKey] as number));
  const front: ModelCard[] = [];
  let best = -1;
  for (const c of sorted) if ((c.metrics.f1_macro as number) > best) { front.push(c); best = c.metrics.f1_macro as number; }
  const xticks: number[] = [];
  for (let e = Math.ceil(xmin); e <= Math.floor(xmax); e++) xticks.push(e);
  const yticks = Array.from({ length: 6 }, (_, i) => ymin + ((ymax - ymin) * i) / 5);
  return (
    <div ref={ref} style={{ width: "100%" }}>
      <svg width={width} height={H} className="chart">
        {yticks.map((t) => (
          <g key={t}>
            <line className="grid-line" x1={L} x2={width - R} y1={Y(t)} y2={Y(t)} />
            <text x={L - 8} y={Y(t) + 4} textAnchor="end">{t.toFixed(3)}</text>
          </g>
        ))}
        {xticks.map((e) => (
          <g key={e}>
            <line className="grid-line" x1={X(10 ** e)} x2={X(10 ** e)} y1={T} y2={H - B} />
            <text x={X(10 ** e)} y={H - B + 16} textAnchor="middle">{fmt.si(10 ** e)}</text>
          </g>
        ))}
        <text className="axis-label" x={(L + width - R) / 2} y={H - 6} textAnchor="middle">{XOPTS.find((o) => o.id === xKey)!.label} (log scale)</text>
        <text className="axis-label" x={14} y={(T + H - B) / 2} textAnchor="middle" transform={`rotate(-90 14 ${(T + H - B) / 2})`}>test macro-F1</text>
        <motion.path
          key={xKey}
          d={front.map((c, i) => `${i ? "L" : "M"}${X(c.metrics[xKey] as number)},${Y(c.metrics.f1_macro as number)}`).join("")}
          fill="none" stroke="var(--accent)" strokeWidth={1.5} strokeDasharray="5 5"
          initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 1.2 }}
        />
        {pts.map((c, i) => {
          const r = 6 + 16 * Math.sqrt((c.metrics.params ?? 0) / pmax);
          const isC = c.kind === "cascade";
          return (
            <motion.g key={c.id}
              initial={{ opacity: 0, scale: 0 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: i * 0.05, type: "spring" }}
              style={{ transformOrigin: `${X(c.metrics[xKey] as number)}px ${Y(c.metrics.f1_macro as number)}px` }}
              onMouseMove={(e) => tip.show(e, (
                <div>
                  <b>{c.name}</b><br />macro-F1 {fmt.num(c.metrics.f1_macro)} · recall {fmt.num(c.metrics.recall_malicious)}<br />
                  {fmt.si(c.metrics.params)} params · {fmt.si(c.metrics.macs)} MACs{isC ? " (expected)" : ""}<br />
                  {fmt.num(c.metrics.size_kb, 1)} KB · {fmt.ms(c.metrics.latency_p50_ms)}
                </div>
              ))}
              onMouseLeave={tip.hide}
            >
              <motion.circle
                cx={X(c.metrics[xKey] as number)} cy={Y(c.metrics.f1_macro as number)} r={r}
                fill={isC ? "var(--accent)" : "var(--line-2)"} fillOpacity={isC ? 0.85 : 0.6}
                stroke="var(--panel)" strokeWidth={2}
                animate={{ cx: X(c.metrics[xKey] as number), cy: Y(c.metrics.f1_macro as number) }}
                transition={{ type: "spring", stiffness: 80, damping: 16 }}
              />
              <motion.text
                animate={{ x: X(c.metrics[xKey] as number) + r + 4, y: Y(c.metrics.f1_macro as number) + 4 }}
                transition={{ type: "spring", stiffness: 80, damping: 16 }}
                style={{ fill: isC ? "var(--text)" : "var(--text-2)", fontSize: 11.5, fontWeight: isC ? 650 : 400 }}
              >
                {c.name.replace(/^B\d+ /, (m) => m)}
              </motion.text>
            </motion.g>
          );
        })}
      </svg>
      {tip.node}
    </div>
  );
}

type Col = { id: string; label: string; get: (c: ModelCard) => number | null | undefined; f: (v: number | null | undefined) => string };
const COLS: Col[] = [
  { id: "f1", label: "macro-F1", get: (c) => c.metrics.f1_macro, f: (v) => fmt.num(v) },
  { id: "rec", label: "Recall", get: (c) => c.metrics.recall_malicious, f: (v) => fmt.num(v) },
  { id: "fpr", label: "FPR", get: (c) => c.metrics.fpr, f: (v) => fmt.num(v, 4) },
  { id: "params", label: "Params", get: (c) => c.metrics.params, f: fmt.int },
  { id: "macs", label: "MACs", get: (c) => c.metrics.macs, f: fmt.si },
  { id: "size", label: "Size KB", get: (c) => c.metrics.size_kb, f: (v) => fmt.num(v, 1) },
  { id: "lat", label: "p50 latency", get: (c) => c.metrics.latency_p50_ms, f: fmt.ms },
];

export function BenchmarksPage() {
  const { cards } = useLive();
  const [xKey, setX] = useState<XKey>("macs");
  const [sort, setSort] = useState<{ id: string; dir: 1 | -1 }>({ id: "f1", dir: -1 });
  const rows = useMemo(() => {
    const col = COLS.find((c) => c.id === sort.id)!;
    return [...cards].sort((a, b) => ((col.get(a) ?? -Infinity) - (col.get(b) ?? -Infinity)) * sort.dir);
  }, [cards, sort]);
  const synthetic = cards.some((c) => c.is_synthetic);
  const maxSize = Math.max(...cards.map((c) => c.metrics.size_kb ?? 0));
  const maxLat = Math.max(...cards.map((c) => c.metrics.latency_p50_ms ?? 0));

  return (
    <>
      <div className="page-head">
        <div>
          <h2>Benchmarks</h2>
          <p>
            Stored test-split results of every model: held-out devices, same leakage-safe pipeline for all. For LiteCascade, MACs are
            the expected cost E[C] = C₁ + (1 − ρ)·C₂ at its measured exit rate. Bubble size = parameters.
          </p>
        </div>
        {synthetic && <span className="badge syn">synthetic fixture data, not real-world performance</span>}
      </div>
      <div className="panel" style={{ marginBottom: 16 }}>
        <div className="panel-head">
          <h3>Accuracy vs. cost · dashed line = Pareto front</h3>
          <div className="seg">
            {XOPTS.map((o) => (
              <button key={o.id} className={xKey === o.id ? "on" : ""} onClick={() => setX(o.id)}>
                {xKey === o.id && <motion.span layoutId="x-bg" className="seg-bg" />}
                {o.label}
              </button>
            ))}
          </div>
        </div>
        <Pareto cards={cards} xKey={xKey} />
      </div>
      <div className="grid g2">
        <div className="panel">
          <div className="panel-head"><h3>Model size (serialized)</h3></div>
          {[...cards].sort((a, b) => (a.metrics.size_kb ?? 0) - (b.metrics.size_kb ?? 0)).map((c) => (
            <div key={c.id} style={{ marginBottom: 8 }}>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12.5 }}><span>{c.name}</span><span className="num">{fmt.num(c.metrics.size_kb, 1)} KB</span></div>
              <Bar value={c.metrics.size_kb ?? 0} max={maxSize} color={c.kind === "cascade" ? "var(--accent)" : "var(--line-2)"} />
            </div>
          ))}
        </div>
        <div className="panel">
          <div className="panel-head"><h3>Latency p50, batch 1, host CPU</h3></div>
          {[...cards].sort((a, b) => (a.metrics.latency_p50_ms ?? 0) - (b.metrics.latency_p50_ms ?? 0)).map((c) => (
            <div key={c.id} style={{ marginBottom: 8 }}>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12.5 }}><span>{c.name}</span><span className="num">{fmt.ms(c.metrics.latency_p50_ms)}</span></div>
              <Bar value={c.metrics.latency_p50_ms ?? 0} max={maxLat} color={c.kind === "cascade" ? "var(--accent)" : "var(--line-2)"} />
            </div>
          ))}
          <p className="hint">PyTorch eager on this PC. INT8 kernels add per-op overhead here, so INT8 is not faster on this CPU; no Raspberry Pi or MCU was measured.</p>
        </div>
      </div>
      <div className="panel" style={{ marginTop: 16 }}>
        <div className="panel-head"><h3>All models</h3><span className="hint">click a column to sort</span></div>
        <div className="scroll">
          <table className="t">
            <thead>
              <tr>
                <th>Model</th><th>Family</th>
                {COLS.map((c) => (
                  <th key={c.id} className="sort" onClick={() => setSort({ id: c.id, dir: sort.id === c.id ? ((-sort.dir) as 1 | -1) : -1 })}>
                    {c.label}{sort.id === c.id ? (sort.dir === -1 ? " ↓" : " ↑") : ""}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((c) => (
                <motion.tr key={c.id} layout className={c.kind === "cascade" ? "hl" : ""}>
                  <td><b>{c.name}</b></td><td className="sub">{c.family}</td>
                  {COLS.map((col) => <td key={col.id}>{col.f(col.get(c))}</td>)}
                </motion.tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </>
  );
}
