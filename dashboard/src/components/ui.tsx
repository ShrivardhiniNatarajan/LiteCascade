import { animate, motion, useMotionValue, useTransform } from "framer-motion";
import { useEffect, useRef, useState, type ReactNode } from "react";

/** Number that tweens to its new value. */
export function AnimatedNumber({ value, format }: { value: number | null | undefined; format: (v: number) => string }) {
  const mv = useMotionValue(value ?? 0);
  const [text, setText] = useState(value == null ? "–" : format(value));
  useEffect(() => {
    if (value == null) {
      setText("–");
      return;
    }
    const c = animate(mv, value, { duration: 0.6, ease: "easeOut", onUpdate: (v) => setText(format(v)) });
    return () => c.stop();
  }, [value, format, mv]);
  return <span className="num">{text}</span>;
}

export function Tile({ k, v, d, accent, children }: { k: string; v: ReactNode; d?: ReactNode; accent?: string; children?: ReactNode }) {
  return (
    <motion.div className="tile" layout initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}>
      {accent && <div style={{ position: "absolute", inset: "0 auto 0 0", width: 3, background: accent }} />}
      <div className="k">{k}</div>
      <div className="v">{v}</div>
      {d && <div className="d">{d}</div>}
      {children}
    </motion.div>
  );
}

/** Circular gauge (0..1). */
export function Ring({ value, size = 92, stroke = 9, color = "var(--accent)", label }: {
  value: number | null | undefined; size?: number; stroke?: number; color?: string; label?: ReactNode;
}) {
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const v = Math.max(0, Math.min(1, value ?? 0));
  return (
    <div style={{ position: "relative", width: size, height: size }}>
      <svg width={size} height={size} style={{ transform: "rotate(-90deg)" }}>
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="var(--line)" strokeWidth={stroke} />
        <motion.circle
          cx={size / 2} cy={size / 2} r={r} fill="none" stroke={color} strokeWidth={stroke} strokeLinecap="round"
          strokeDasharray={c} animate={{ strokeDashoffset: c * (1 - v) }} transition={{ type: "spring", stiffness: 60, damping: 18 }}
        />
      </svg>
      <div style={{ position: "absolute", inset: 0, display: "grid", placeItems: "center", textAlign: "center", fontWeight: 650 }}>
        {label}
      </div>
    </div>
  );
}

export function Sparkline({ values, width = 120, height = 32, color = "var(--accent)" }: {
  values: number[]; width?: number; height?: number; color?: string;
}) {
  if (!values.length) return <svg width={width} height={height} />;
  const max = Math.max(...values) * 1.1 || 1;
  const min = Math.min(...values) * 0.9;
  const xs = (i: number) => (values.length === 1 ? width : (i / (values.length - 1)) * width);
  const ys = (v: number) => height - 2 - ((v - min) / (max - min || 1)) * (height - 4);
  const d = values.map((v, i) => `${i ? "L" : "M"}${xs(i).toFixed(1)},${ys(v).toFixed(1)}`).join("");
  const area = `${d}L${width},${height}L0,${height}Z`;
  return (
    <svg width={width} height={height} style={{ display: "block", overflow: "visible" }}>
      <path d={area} fill={color} opacity={0.12} />
      <path d={d} fill="none" stroke={color} strokeWidth={1.8} strokeLinejoin="round" />
      <circle cx={xs(values.length - 1)} cy={ys(values[values.length - 1])} r={2.8} fill={color} />
    </svg>
  );
}

export function Bar({ value, max, color = "var(--accent)", height = 8 }: { value: number; max: number; color?: string; height?: number }) {
  return (
    <div style={{ height, background: "var(--line)", borderRadius: 99, overflow: "hidden" }}>
      <motion.div
        style={{ height: "100%", background: color, borderRadius: 99 }}
        initial={{ width: 0 }} animate={{ width: `${Math.max(0, Math.min(1, value / (max || 1))) * 100}%` }}
        transition={{ type: "spring", stiffness: 70, damping: 18 }}
      />
    </div>
  );
}

/** Small fixed tooltip that follows the pointer. */
export function useTooltip() {
  const [tip, setTip] = useState<{ x: number; y: number; html: ReactNode } | null>(null);
  const node = tip ? (
    <div className="tooltip" style={{ left: Math.min(tip.x + 14, window.innerWidth - 300), top: tip.y + 14 }}>{tip.html}</div>
  ) : null;
  return {
    node,
    show: (e: { clientX: number; clientY: number }, html: ReactNode) => setTip({ x: e.clientX, y: e.clientY, html }),
    hide: () => setTip(null),
  };
}

export function useElementWidth<T extends HTMLElement>(fallback = 600) {
  const ref = useRef<T>(null);
  const [w, setW] = useState(fallback);
  useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(200, e.contentRect.width)));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  return [ref, w] as const;
}

export function Verdict({ pred }: { pred: number }) {
  return pred > 0 ? <span className="badge mal">malicious</span> : <span className="badge ben">benign</span>;
}

export const pulse = (seed: number) => ({
  key: seed,
  initial: { scale: 1.6, opacity: 0.9 },
  animate: { scale: 1, opacity: 1 },
  transition: { type: "spring" as const, stiffness: 300, damping: 15 },
});

export { useTransform };
