import { AnimatePresence, motion } from "framer-motion";
import { Pause, Play, RotateCcw, ShieldAlert, SkipForward, Zap } from "lucide-react";
import { useMemo } from "react";
import { Pipeline } from "../components/Pipeline";
import { ScoreStream } from "../components/ScoreStream";
import { AnimatedNumber, Ring, Tile } from "../components/ui";
import { fmt } from "../lib/api";
import { useLive } from "../lib/live";

const SPEEDS = [1, 5, 20, 100];

export function Controls() {
  const { running, speed, control, setSpeed } = useLive();
  return (
    <div className="toolbar">
      <motion.button whileTap={{ scale: 0.94 }} className={`btn ${running ? "" : "primary"}`} onClick={() => control(running ? "pause" : "play")}>
        {running ? <Pause size={16} /> : <Play size={16} />} {running ? "Pause" : "Start stream"}
      </motion.button>
      <motion.button whileTap={{ scale: 0.94 }} className="btn" onClick={() => control("step")} disabled={running} title="Replay one window">
        <SkipForward size={16} /> Step
      </motion.button>
      <motion.button whileTap={{ scale: 0.94 }} className="btn" onClick={() => control("reset")} title="Reset counters">
        <RotateCcw size={16} />
      </motion.button>
      <div className="seg" role="group" aria-label="Replay speed">
        {SPEEDS.map((s) => (
          <button key={s} className={speed === s ? "on" : ""} onClick={() => setSpeed(s)}>
            {speed === s && <motion.span layoutId="speed-bg" className="seg-bg" />}
            {s}×
          </button>
        ))}
      </div>
    </div>
  );
}

export function InjectBar() {
  const { families, inject } = useLive();
  return (
    <div className="toolbar">
      <span className="hint" style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
        <ShieldAlert size={14} /> Inject a burst of real held-out attack windows:
      </span>
      {families.map((f) => (
        <motion.button key={f} whileTap={{ scale: 0.92 }} whileHover={{ y: -1 }} className="chip" onClick={() => inject(f, 30)}>
          <Zap size={12} /> {fmt.family(f)}
        </motion.button>
      ))}
    </div>
  );
}

export function LivePage() {
  const { events, summary, primary, cardsById, speed, last } = useLive();
  const prim = primary ? summary?.models[primary] : undefined;
  const card = primary ? cardsById[primary] : undefined;
  const tauB = card?.gate?.tau_b as number | undefined;

  // Alert state: majority of the last 8 primary verdicts malicious
  const recent = events.slice(-8);
  const malShare = primary && recent.length ? recent.filter((e) => (e.outputs[primary]?.pred ?? 0) > 0).length / recent.length : 0;
  const alert = malShare >= 0.5;
  const feed = useMemo(() => events.slice(-12).reverse(), [events]);

  return (
    <>
      <div className="page-head">
        <div>
          <h2>Live cascade</h2>
          <p>
            Held-out windows from devices the models never saw are replayed through LiteCascade and every other model. Each
            packet below is one window; its path shows whether the Sentinel released it or the Analyst had to run.
          </p>
        </div>
        <Controls />
      </div>

      <motion.div
        className="status"
        animate={{
          borderColor: alert ? "var(--malicious)" : "var(--line)",
          boxShadow: alert ? "0 0 0 1px var(--malicious), 0 0 32px -8px var(--malicious)" : "0 0 0 0 transparent",
        }}
        transition={{ duration: 0.4 }}
      >
        <motion.span
          className="dot"
          style={{ background: alert ? "var(--malicious)" : "var(--ok)" }}
          animate={{ scale: [1, 1.5, 1], opacity: [1, 0.6, 1] }}
          transition={{ repeat: Infinity, duration: alert ? 0.7 : 1.8 }}
        />
        <motion.strong key={alert ? "a" : "n"} initial={{ y: 8, opacity: 0 }} animate={{ y: 0, opacity: 1 }}>
          {alert ? "INTRUSION DETECTED" : "TRAFFIC NORMAL"}
        </motion.strong>
        <span className="sub" style={{ marginLeft: "auto" }}>
          {last ? (
            <>
              window #{last.step} · {last.segment} · ground truth: <b>{fmt.family(last.family)}</b>
            </>
          ) : (
            "Press Start stream to begin the replay"
          )}
        </span>
      </motion.div>

      <div className="tiles">
        <Tile k="Windows replayed" v={<AnimatedNumber value={summary?.steps ?? 0} format={(v) => Math.round(v).toLocaleString()} />} d={`${speed}× replay`} />
        <Tile k="Sentinel exit rate" v={<AnimatedNumber value={summary?.exit_rate ?? null} format={(v) => `${(v * 100).toFixed(1)}%`} />}
          d="windows that skipped the Analyst" accent="var(--sentinel)" />
        <Tile k="Compute saved" v={<AnimatedNumber value={summary?.compute_saved ?? null} format={(v) => `${(v * 100).toFixed(1)}%`} />}
          d="MACs vs. always running both exits" accent="var(--ok)" />
        <Tile k="Alerts raised" v={<AnimatedNumber value={summary?.alerts ?? 0} format={(v) => Math.round(v).toLocaleString()} />} d="LiteCascade verdict = malicious" accent="var(--malicious)" />
        <Tile k="Live macro-F1" v={<AnimatedNumber value={prim?.f1_macro ?? null} format={(v) => v.toFixed(3)} />} d={`${card?.name ?? ""}, scored vs. ground truth`} />
        <Tile k="Latency / window" v={fmt.ms(prim?.latency_ms_recent)} d="measured here, batch 1, host CPU" />
      </div>

      <div className="grid g-live">
        <div className="panel">
          <div className="panel-head">
            <h3>Cascade pipeline · {card?.name ?? "LiteCascade"}</h3>
            <span className="hint">packet colour = model verdict</span>
          </div>
          <Pipeline events={events} summary={summary} primary={primary} tauB={tauB} speed={speed} />
        </div>
        <div className="panel" style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <div className="panel-head"><h3>Gate telemetry</h3></div>
          <div style={{ display: "flex", gap: 18, alignItems: "center", flexWrap: "wrap" }}>
            <Ring value={summary?.exit_rate} color="var(--sentinel)" label={<div><div style={{ fontSize: 18 }}>{fmt.pct(summary?.exit_rate ?? null, 0)}</div><div className="hint">exit</div></div>} />
            <Ring value={summary?.compute_saved} color="var(--ok)" label={<div><div style={{ fontSize: 18 }}>{fmt.pct(summary?.compute_saved ?? null, 0)}</div><div className="hint">saved</div></div>} />
            <Ring value={prim?.recall} color="var(--malicious)" label={<div><div style={{ fontSize: 18 }}>{fmt.pct(prim?.recall ?? null, 0)}</div><div className="hint">recall</div></div>} />
          </div>
          <div className="hint">
            Gate: exit benign if p₀ ≥ τ<sub>b</sub> = {tauB != null ? tauB.toFixed(4) : "–"} (Clopper–Pearson, ε = {String(card?.gate?.epsilon ?? "–")},
            α = {String(card?.gate?.alpha ?? "–")}); exit an alert if max p ≥ τ<sub>m</sub> = {card?.gate?.tau_m != null ? Number(card.gate.tau_m).toFixed(4) : "–"}.
          </div>
          <InjectBar />
        </div>
      </div>

      <div className="grid g-live" style={{ marginTop: 16 }}>
        <div className="panel">
          <div className="panel-head">
            <h3>Score stream</h3>
            <div className="toolbar hint">
              <span><span className="badge sen">● Sentinel exit</span></span>
              <span><span className="badge ana">● escalated</span></span>
              <span>— final p(malicious)</span>
              <span style={{ color: "var(--malicious)" }}>▮ true attack (for scoring)</span>
            </div>
          </div>
          <ScoreStream events={events} primary={primary} />
        </div>
        <div className="panel">
          <div className="panel-head"><h3>Event feed</h3><span className="hint">newest first</span></div>
          <div className="feed">
            <AnimatePresence initial={false}>
              {feed.map((e) => {
                const o = primary ? e.outputs[primary] : undefined;
                if (!o) return null;
                const wrong = (o.pred > 0 ? 1 : 0) !== e.label;
                return (
                  <motion.div key={e.step} layout className="feed-row"
                    initial={{ opacity: 0, x: 30, backgroundColor: o.pred > 0 ? "rgba(230,103,103,.25)" : "rgba(57,135,229,.18)" }}
                    animate={{ opacity: 1, x: 0, backgroundColor: "var(--panel-2)" }}
                    exit={{ opacity: 0, height: 0 }} transition={{ duration: 0.35 }}>
                    <span className="mono hint">#{e.step}</span>
                    <span style={{ overflow: "hidden", textOverflow: "ellipsis" }}>
                      {fmt.family(e.family)} {wrong && <span title="prediction differs from ground truth" style={{ color: "var(--warn)" }}>⚠</span>}
                    </span>
                    {o.pred > 0 ? <span className="badge mal">alert</span> : <span className="badge ben">benign</span>}
                    {o.exit ? <span className="badge sen">sentinel</span> : <span className="badge ana">analyst</span>}
                  </motion.div>
                );
              })}
            </AnimatePresence>
          </div>
        </div>
      </div>
    </>
  );
}
