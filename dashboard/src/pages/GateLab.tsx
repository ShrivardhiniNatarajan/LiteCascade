import { motion } from "framer-motion";
import { useEffect, useMemo, useState } from "react";
import { AnimatedNumber, Tile, useElementWidth, useTooltip } from "../components/ui";
import { api, fmt, type GatePoint } from "../lib/api";

type Curve = { model: string; deployed: Record<string, number | boolean>; curve: GatePoint[] };
const ALPHAS = [0.01, 0.05, 0.1];

function CurveChart({ curve, current }: { curve: GatePoint[]; current: GatePoint | null }) {
  const [ref, width] = useElementWidth<HTMLDivElement>(700);
  const tip = useTooltip();
  const H = 320, L = 50, R = 16, T = 14, B = 40;
  const lx = (e: number) => Math.log10(e);
  const x0 = lx(curve[0]?.epsilon ?? 0.001), x1 = lx(curve[curve.length - 1]?.epsilon ?? 0.1);
  const X = (e: number) => L + ((lx(e) - x0) / (x1 - x0 || 1)) * (width - L - R);
  const Y = (v: number) => T + (1 - v) * (H - T - B);
  const Ym = (v: number) => T + (1 - Math.min(v, 0.1) / 0.1) * (H - T - B); // miss-rate axis 0..10%
  const line = (f: (p: GatePoint) => number, y: (v: number) => number) =>
    curve.map((p, i) => `${i ? "L" : "M"}${X(p.epsilon).toFixed(1)},${y(f(p)).toFixed(1)}`).join("");
  const firstFeasible = curve.find((p) => p.feasible);
  return (
    <div ref={ref} style={{ width: "100%" }}>
      <svg width={width} height={H} className="chart">
        {firstFeasible && firstFeasible.epsilon > curve[0].epsilon && (
          <rect x={L} y={T} width={X(firstFeasible.epsilon) - L} height={H - T - B} fill="var(--warn)" opacity={0.08} />
        )}
        {[0, 0.25, 0.5, 0.75, 1].map((v) => (
          <g key={v}>
            <line className="grid-line" x1={L} x2={width - R} y1={Y(v)} y2={Y(v)} />
            <text x={L - 8} y={Y(v) + 4} textAnchor="end">{(v * 100).toFixed(0)}%</text>
            <text x={width - R} y={Y(v) - 3} textAnchor="end" style={{ fill: "var(--malicious)", fontSize: 10 }}>{(v * 10).toFixed(1)}%</text>
          </g>
        ))}
        {curve.map((p) => (
          [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1].some((t) => Math.abs(t - p.epsilon) < 1e-9) && (
            <text key={p.epsilon} x={X(p.epsilon)} y={H - B + 16} textAnchor="middle">
              {+(p.epsilon * 100).toFixed(1)}%
            </text>
          )
        ))}
        <text className="axis-label" x={(L + width) / 2} y={H - 4} textAnchor="middle">risk budget ε (log scale)</text>
        {/* bound: miss rate must stay below epsilon */}
        <path d={line((p) => p.epsilon, Ym)} fill="none" stroke="var(--malicious)" strokeDasharray="3 5" strokeWidth={1.2} />
        <motion.path d={line((p) => p.rho, Y)} fill="none" stroke="var(--sentinel)" strokeWidth={2.6}
          initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 1 }} />
        <motion.path d={line((p) => p.f1_macro, Y)} fill="none" stroke="var(--text)" strokeWidth={1.6}
          initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 1, delay: 0.2 }} />
        <motion.path d={line((p) => p.exit_induced_miss_rate, Ym)} fill="none" stroke="var(--malicious)" strokeWidth={2.2}
          initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 1, delay: 0.4 }} />
        {curve.map((p) => (
          <circle key={p.epsilon} cx={X(p.epsilon)} cy={Y(p.rho)} r={10} fill="transparent"
            onMouseMove={(e) => tip.show(e, <div><b>ε = {p.epsilon}</b><br />ρ = {fmt.pct(p.rho)} · F1 {fmt.num(p.f1_macro)}<br />exit-induced miss {fmt.pct(p.exit_induced_miss_rate, 2)}<br />{p.feasible ? `τb = ${p.tau_b.toFixed(4)}` : "infeasible: benign exit disabled"}</div>)}
            onMouseLeave={tip.hide} />
        ))}
        {current && (
          <motion.g animate={{ x: X(current.epsilon) }} initial={false} transition={{ type: "spring", stiffness: 120, damping: 18 }}>
            <line x1={0} x2={0} y1={T} y2={H - B} stroke="var(--accent)" strokeWidth={1.5} />
            <circle cx={0} cy={Y(current.rho)} r={6} fill="var(--sentinel)" stroke="var(--panel)" strokeWidth={2} style={{ transition: "cy .35s ease" }} />
            <circle cx={0} cy={Ym(current.exit_induced_miss_rate)} r={5} fill="var(--malicious)" stroke="var(--panel)" strokeWidth={2}
              style={{ transition: "cy .35s ease" }} />
          </motion.g>
        )}
      </svg>
      {tip.node}
    </div>
  );
}

export function GateLabPage() {
  const [data, setData] = useState<Curve | null>(null);
  const [alpha, setAlpha] = useState(0.05);
  const [epsIdx, setEpsIdx] = useState(50);
  const [point, setPoint] = useState<GatePoint | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const eps = useMemo(() => 10 ** (-3 + (epsIdx / 100) * 2), [epsIdx]); // 0.1% .. 10%

  useEffect(() => {
    api.get<Curve>(`/api/gate/curve?alpha=${alpha}`).then(setData).catch((e) => setErr(e.message));
  }, [alpha]);
  useEffect(() => {
    const t = setTimeout(() => {
      api.get<GatePoint>(`/api/gate/point?epsilon=${eps}&alpha=${alpha}`).then(setPoint).catch((e) => setErr(e.message));
    }, 120);
    return () => clearTimeout(t);
  }, [eps, alpha]);

  return (
    <>
      <div className="page-head">
        <div>
          <h2>Gate lab</h2>
          <p>
            The Sentinel may release a window as benign only if p₀ ≥ τ<sub>b</sub>. τ<sub>b</sub> is the smallest threshold whose one-sided
            Clopper–Pearson upper bound on exit-induced misses stays ≤ ε with confidence 1 − α. Every value here is recomputed from the
            model's real calibration outputs and then measured on the held-out test devices.
          </p>
        </div>
      </div>
      {err && <div className="status" style={{ borderColor: "var(--malicious)" }}>{err}</div>}
      <div className="panel" style={{ marginBottom: 16 }}>
        <div className="panel-head">
          <h3>Choose the risk budget</h3>
          <div className="seg">
            {ALPHAS.map((a) => (
              <button key={a} className={alpha === a ? "on" : ""} onClick={() => setAlpha(a)}>
                {alpha === a && <motion.span layoutId="alpha-bg" className="seg-bg" />}α = {a}
              </button>
            ))}
          </div>
        </div>
        <div style={{ display: "flex", gap: 16, alignItems: "center" }}>
          <span className="mono" style={{ minWidth: 110 }}>ε = {(eps * 100).toFixed(2)}%</span>
          <input type="range" min={0} max={100} value={epsIdx} onChange={(e) => setEpsIdx(Number(e.target.value))} aria-label="epsilon" />
        </div>
      </div>
      <div className="tiles">
        <Tile k="τb (benign exit)" v={<motion.span key={String(point?.feasible)} initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
          {point ? (point.feasible ? point.tau_b.toFixed(4) : "disabled") : "–"}</motion.span>}
          d={point && !point.feasible ? `needs ≥ ${point.min_N_m} malicious calib. windows, has ${point.N_m}` : `N_m = ${point?.N_m ?? "–"}, misses allowed ≤ ${point?.m_max ?? "–"}`}
          accent="var(--warn)" />
        <Tile k="Exit rate ρ (test)" v={<AnimatedNumber value={point?.rho} format={(v) => `${(v * 100).toFixed(1)}%`} />} accent="var(--sentinel)" d="windows that skip the Analyst" />
        <Tile k="Exit-induced misses" v={<AnimatedNumber value={point?.exit_induced_miss_rate} format={(v) => `${(v * 100).toFixed(2)}%`} />}
          accent="var(--malicious)" d={point ? (point.exit_induced_miss_rate <= point.epsilon ? "within ε on test ✓" : "above ε on test") : ""} />
        <Tile k="Cascade macro-F1" v={<AnimatedNumber value={point?.f1_macro} format={(v) => v.toFixed(4)} />} d={`recall ${fmt.num(point?.recall)} · FPR ${fmt.num(point?.fpr, 4)}`} />
      </div>
      <div className="panel">
        <div className="panel-head">
          <h3>Efficiency vs. risk</h3>
          <div className="toolbar hint">
            <span style={{ color: "var(--sentinel)" }}>━ exit rate ρ</span>
            <span>━ macro-F1</span>
            <span style={{ color: "var(--malicious)" }}>━ exit-induced miss rate (right axis)</span>
            <span style={{ color: "var(--malicious)" }}>┄ bound ε</span>
            <span style={{ color: "var(--warn)" }}>▮ infeasible</span>
          </div>
        </div>
        {data && <CurveChart curve={data.curve} current={point} />}
        <p className="hint">
          Deployed gate of {data?.model}: τ<sub>b</sub> = {data ? Number(data.deployed.tau_b).toFixed(4) : "–"} at ε = {String(data?.deployed.epsilon ?? "–")}, α ={" "}
          {String(data?.deployed.alpha ?? "–")}. The guarantee assumes calibration and test windows are exchangeable; it bounds the extra misses caused
          by exiting early, not the Analyst's own misses.
        </p>
      </div>
    </>
  );
}
