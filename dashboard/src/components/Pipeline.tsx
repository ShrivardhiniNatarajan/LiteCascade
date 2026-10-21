import { AnimatePresence, motion } from "framer-motion";
import { fmt, type LiveEvent, type LiveSummary } from "../lib/api";

type P = { x: number; y: number };
const N: Record<string, P> = {
  in: { x: 54, y: 140 },
  pip: { x: 178, y: 140 },
  stem: { x: 298, y: 140 },
  sen: { x: 422, y: 140 },
  gate: { x: 548, y: 140 },
  early: { x: 700, y: 62 },
  ana: { x: 700, y: 218 },
  out: { x: 846, y: 140 },
};
const EXIT_ROUTE = ["in", "pip", "stem", "sen", "gate", "early", "out"];
const FULL_ROUTE = ["in", "pip", "stem", "sen", "gate", "ana", "out"];

function Node({ id, label, sub, color, w = 104, glow }: { id: string; label: string; sub?: string; color: string; w?: number; glow?: number }) {
  const p = N[id];
  return (
    <g>
      <motion.rect
        key={glow}
        x={p.x - w / 2} y={p.y - 30} width={w} height={60} rx={14}
        fill="var(--panel-2)" stroke={color} strokeWidth={1.6}
        initial={{ filter: "drop-shadow(0 0 0px transparent)" }}
        animate={{ filter: ["drop-shadow(0 0 12px " + color + ")", "drop-shadow(0 0 0px transparent)"] }}
        transition={{ duration: 0.7 }}
      />
      <text x={p.x} y={p.y - 4} textAnchor="middle" style={{ fill: "var(--text)", fontSize: 13, fontWeight: 650 }}>{label}</text>
      {sub && <text x={p.x} y={p.y + 15} textAnchor="middle" style={{ fill: "var(--text-2)", fontSize: 11 }}>{sub}</text>}
    </g>
  );
}

function Gate({ glow, tau }: { glow?: number; tau?: number }) {
  const p = N.gate;
  return (
    <g>
      <motion.path
        key={glow}
        d={`M${p.x} ${p.y - 40} L${p.x + 44} ${p.y} L${p.x} ${p.y + 40} L${p.x - 44} ${p.y} Z`}
        fill="var(--panel-2)" stroke="var(--warn)" strokeWidth={1.6}
        animate={{ filter: ["drop-shadow(0 0 10px var(--warn))", "drop-shadow(0 0 0px transparent)"] }}
        transition={{ duration: 0.6 }}
      />
      <text x={p.x} y={p.y - 2} textAnchor="middle" style={{ fill: "var(--text)", fontSize: 12.5, fontWeight: 650 }}>CP gate</text>
      <text x={p.x} y={p.y + 14} textAnchor="middle" style={{ fill: "var(--text-2)", fontSize: 10.5 }}>
        τb={tau != null && tau <= 1 ? tau.toFixed(3) : "off"}
      </text>
    </g>
  );
}

const edge = (a: string, b: string, dashed = false, color = "var(--line-2)") => {
  const p = N[a], q = N[b];
  const mx = (p.x + q.x) / 2;
  return (
    <path
      key={a + b}
      d={`M${p.x} ${p.y} C${mx} ${p.y}, ${mx} ${q.y}, ${q.x} ${q.y}`}
      fill="none" stroke={color} strokeWidth={2} strokeDasharray={dashed ? "5 6" : undefined}
    />
  );
};

function bezierPoints(route: string[], steps = 6): P[] {
  const pts: P[] = [];
  for (let i = 0; i < route.length - 1; i++) {
    const p = N[route[i]], q = N[route[i + 1]];
    const mx = (p.x + q.x) / 2;
    for (let s = 0; s < steps; s++) {
      const t = s / steps;
      const u = 1 - t;
      // cubic bezier with control points (mx, p.y) and (mx, q.y)
      const x = u * u * u * p.x + 3 * u * u * t * mx + 3 * u * t * t * mx + t * t * t * q.x;
      const y = u * u * u * p.y + 3 * u * u * t * p.y + 3 * u * t * t * q.y + t * t * t * q.y;
      pts.push({ x, y });
    }
  }
  pts.push(N[route[route.length - 1]]);
  return pts;
}
const EXIT_PTS = bezierPoints(EXIT_ROUTE);
const FULL_PTS = bezierPoints(FULL_ROUTE);

export function Pipeline({ events, summary, primary, tauB, speed }: {
  events: LiveEvent[]; summary: LiveSummary | null; primary: string | null; tauB?: number; speed: number;
}) {
  const recent = events.slice(-14);
  const last = events[events.length - 1];
  const lo = last && primary ? last.outputs[primary] : undefined;
  const dur = Math.max(0.7, 2.4 / Math.sqrt(Math.max(speed, 1)));
  const glow = last?.step;
  return (
    <svg viewBox="0 0 900 280" className="chart" style={{ width: "100%", height: "auto", display: "block" }}>
      <defs>
        <radialGradient id="pkt-b"><stop offset="0%" stopColor="#fff" /><stop offset="60%" stopColor="var(--benign)" /></radialGradient>
        <radialGradient id="pkt-m"><stop offset="0%" stopColor="#fff" /><stop offset="60%" stopColor="var(--malicious)" /></radialGradient>
      </defs>
      {edge("in", "pip")}
      {edge("pip", "stem")}
      {edge("stem", "sen")}
      {edge("sen", "gate")}
      {edge("gate", "early", false, "color-mix(in srgb, var(--sentinel) 55%, var(--line-2))")}
      {edge("early", "out", false, "color-mix(in srgb, var(--sentinel) 55%, var(--line-2))")}
      {edge("gate", "ana", true, "color-mix(in srgb, var(--analyst) 55%, var(--line-2))")}
      {edge("ana", "out", true, "color-mix(in srgb, var(--analyst) 55%, var(--line-2))")}

      <Node id="in" label="Traffic" sub="T=10 records" color="var(--line-2)" w={92} glow={glow} />
      <Node id="pip" label="PIP layer" sub="z = W(x − μ)" color="var(--accent)" glow={glow} />
      <Node id="stem" label="DS stem" sub="dw-sep conv" color="var(--accent)" glow={glow} />
      <Node id="sen" label="Sentinel" sub={lo?.sentinel_score != null ? `p(mal)=${lo.sentinel_score.toFixed(2)}` : "tiny exit"} color="var(--sentinel)" glow={glow} />
      <Gate glow={glow} tau={tauB} />
      <Node id="early" label="Early exit" sub={`ρ = ${fmt.pct(summary?.exit_rate ?? null)}`} color="var(--sentinel)" w={118}
        glow={lo?.exit ? glow : undefined} />
      <Node id="ana" label="Analyst" sub={`${fmt.int(summary?.analyst_calls ?? 0)} calls`} color="var(--analyst)" w={118}
        glow={lo && !lo.exit ? glow : undefined} />
      <Node id="out" label="Verdict" sub={lo ? (lo.pred > 0 ? "MALICIOUS" : "benign") : "–"} color={lo && lo.pred > 0 ? "var(--malicious)" : "var(--benign)"} w={96} glow={glow} />

      <AnimatePresence>
        {primary &&
          recent.map((e) => {
            const o = e.outputs[primary];
            if (!o) return null;
            const pts = o.exit ? EXIT_PTS : FULL_PTS;
            return (
              <motion.circle
                key={e.step}
                cx={pts[0].x}
                cy={pts[0].y}
                r={o.pred > 0 ? 7 : 5.5}
                fill={`url(#pkt-${o.pred > 0 ? "m" : "b"})`}
                initial={{ cx: pts[0].x, cy: pts[0].y, opacity: 0 }}
                animate={{ cx: pts.map((p) => p.x), cy: pts.map((p) => p.y), opacity: [0, 1, 1, 1, 0.9, 0] }}
                exit={{ opacity: 0 }}
                transition={{ duration: dur, ease: "linear" }}
              />
            );
          })}
      </AnimatePresence>
      <text x={700} y={20} textAnchor="middle" style={{ fill: "var(--sentinel)", fontSize: 11 }}>confident → leaves after the Sentinel</text>
      <text x={700} y={270} textAnchor="middle" style={{ fill: "var(--analyst)", fontSize: 11 }}>uncertain → escalated to the Analyst</text>
    </svg>
  );
}
