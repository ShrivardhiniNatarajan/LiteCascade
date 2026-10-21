import { AnimatePresence, motion } from "framer-motion";
import { useMemo, useState } from "react";
import { Bar, Ring, Sparkline } from "../components/ui";
import { fmt, type ModelCard } from "../lib/api";
import { useLive } from "../lib/live";
import { Controls, InjectBar } from "./Live";

const METRICS = [
  { id: "f1_macro", label: "Live macro-F1", better: "high" as const, f: (v: number) => v.toFixed(3) },
  { id: "recall", label: "Malicious recall", better: "high" as const, f: (v: number) => fmt.pct(v) },
  { id: "latency_ms_recent", label: "Latency (ms)", better: "low" as const, f: (v: number) => v.toFixed(2) },
];

const famColor: Record<string, string> = {
  cascade: "var(--accent)",
  hybrid: "var(--analyst)",
  recurrent: "#a78bfa",
  convolutional: "var(--ok)",
  attention: "var(--warn)",
  mlp: "var(--muted)",
};

function ModelTile({ card }: { card: ModelCard }) {
  const { summary, last } = useLive();
  const s = summary?.models[card.id];
  const o = last?.outputs[card.id];
  const wrong = o && last ? (o.pred > 0 ? 1 : 0) !== last.label : false;
  return (
    <motion.div layout className="mcard" initial={{ opacity: 0, scale: 0.96 }} animate={{ opacity: 1, scale: 1 }}
      style={{ borderColor: wrong ? "var(--warn)" : undefined }}>
      <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 3, background: famColor[card.family] ?? "var(--line-2)" }} />
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 8 }}>
        <div style={{ minWidth: 0 }}>
          <div className="name">{card.name}</div>
          <div className="fam">{card.family}{card.quantization !== "none" ? " · INT8" : ""}</div>
        </div>
        <motion.div
          key={last?.step}
          className="led"
          initial={{ scale: 1.8, boxShadow: `0 0 18px ${o && o.pred > 0 ? "var(--malicious)" : "var(--benign)"}` }}
          animate={{ scale: 1, boxShadow: "0 0 0px transparent" }}
          transition={{ duration: 0.5 }}
          style={{ background: o ? (o.pred > 0 ? "var(--malicious)" : "var(--benign)") : "var(--line-2)" }}
          title={o ? `p(malicious) = ${o.score.toFixed(3)}` : "waiting"}
        />
      </div>
      <div style={{ display: "flex", gap: 14, alignItems: "center", marginTop: 10 }}>
        <Ring value={s?.f1_macro} size={70} stroke={7} color={famColor[card.family]}
          label={<span style={{ fontSize: 14 }}>{s ? s.f1_macro.toFixed(2) : "–"}</span>} />
        <div style={{ flex: 1, minWidth: 0, fontSize: 12 }}>
          <div className="hint">p(malicious) now</div>
          <Bar value={o?.score ?? 0} max={1} color={o && o.pred > 0 ? "var(--malicious)" : "var(--benign)"} />
          <div className="hint" style={{ marginTop: 6 }}>latency · {fmt.ms(s?.latency_ms_recent)}</div>
          <Sparkline values={s?.latency_spark ?? []} width={130} height={26} color={famColor[card.family]} />
        </div>
      </div>
      <div className="hint num" style={{ marginTop: 8, display: "flex", justifyContent: "space-between" }}>
        <span>TP {s?.tp ?? 0}</span><span>FP {s?.fp ?? 0}</span><span>FN {s?.fn ?? 0}</span><span>TN {s?.tn ?? 0}</span>
      </div>
      {card.kind === "cascade" && o && (
        <div style={{ marginTop: 6 }}>{o.exit ? <span className="badge sen">Sentinel exit</span> : <span className="badge ana">Analyst ran</span>}</div>
      )}
    </motion.div>
  );
}

export function ArenaPage() {
  const { cards, active, setActive, summary } = useLive();
  const [metric, setMetric] = useState(METRICS[0]);
  const shown = cards.filter((c) => active.includes(c.id));
  const board = useMemo(() => {
    const rows = shown
      .map((c) => ({ c, v: (summary?.models[c.id] as Record<string, unknown> | undefined)?.[metric.id] as number | null | undefined }))
      .filter((r) => r.v != null) as { c: ModelCard; v: number }[];
    rows.sort((a, b) => (metric.better === "high" ? b.v - a.v : a.v - b.v));
    return rows;
  }, [shown, summary, metric]);
  const max = Math.max(...board.map((r) => r.v), 1e-9);
  const toggle = (id: string) => setActive(active.includes(id) ? active.filter((a) => a !== id) : [...active, id]);

  return (
    <>
      <div className="page-head">
        <div>
          <h2>Model arena</h2>
          <p>
            Every deep-learning model receives the same window at the same moment. Scores are running totals against the ground
            truth of the replayed windows; latency is each model's own measured call time on this machine.
          </p>
        </div>
        <Controls />
      </div>
      <div className="panel" style={{ marginBottom: 16 }}>
        <div className="panel-head"><h3>Racing models</h3><span className="hint">click to add or remove · LiteCascade INT8 always runs</span></div>
        <div className="toolbar">
          {cards.map((c) => (
            <motion.button key={c.id} whileTap={{ scale: 0.92 }} className={`chip ${active.includes(c.id) ? "on" : ""}`} onClick={() => toggle(c.id)}>
              <span style={{ width: 8, height: 8, borderRadius: 9, background: famColor[c.family] }} /> {c.name}
            </motion.button>
          ))}
        </div>
        <div style={{ marginTop: 12 }}><InjectBar /></div>
      </div>

      <div className="grid g-live">
        <div className="arena">
          <AnimatePresence>{shown.map((c) => <ModelTile key={c.id} card={c} />)}</AnimatePresence>
        </div>
        <div className="panel" style={{ alignSelf: "start" }}>
          <div className="panel-head">
            <h3>Leaderboard</h3>
            <div className="seg">
              {METRICS.map((m) => (
                <button key={m.id} className={metric.id === m.id ? "on" : ""} onClick={() => setMetric(m)}>
                  {metric.id === m.id && <motion.span layoutId="lb-bg" className="seg-bg" />}
                  {m.label.split(" ")[m.label.split(" ").length - 1].replace(/[()]/g, "")}
                </button>
              ))}
            </div>
          </div>
          <div className="hint" style={{ marginBottom: 10 }}>{metric.label} · {metric.better === "high" ? "higher is better" : "lower is better"}</div>
          {board.length === 0 && <div className="hint">Start the stream to fill the leaderboard.</div>}
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {board.map((r, i) => (
              <motion.div key={r.c.id} layout transition={{ type: "spring", stiffness: 300, damping: 30 }}>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12.5, marginBottom: 3 }}>
                  <span><span className="hint mono">{i + 1}.</span> {r.c.name}</span>
                  <span className="num">{metric.f(r.v)}</span>
                </div>
                <Bar value={r.v} max={max} color={r.c.kind === "cascade" ? "var(--accent)" : "var(--line-2)"} />
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    </>
  );
}
