import { useEffect, useRef } from "react";
import type { LiveEvent } from "../lib/api";
import { useElementWidth } from "./ui";

/** Scrolling strip chart: final malicious score (line) and Sentinel score (dots) of the primary model.
 *  Shaded columns mark windows whose ground truth is malicious (shown for scoring context only). */
export function ScoreStream({ events, primary, height = 220 }: { events: LiveEvent[]; primary: string | null; height?: number }) {
  const [wrap, width] = useElementWidth<HTMLDivElement>();
  const canvas = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const c = canvas.current;
    if (!c || !primary) return;
    const dpr = window.devicePixelRatio || 1;
    c.width = width * dpr;
    c.height = height * dpr;
    const ctx = c.getContext("2d")!;
    ctx.scale(dpr, dpr);
    const css = getComputedStyle(document.documentElement);
    const col = (n: string) => css.getPropertyValue(n).trim();
    const L = 34, R = 8, T = 10, B = 22;
    const iw = width - L - R, ih = height - T - B;
    ctx.clearRect(0, 0, width, height);
    ctx.font = "11px system-ui";
    ctx.fillStyle = col("--muted");
    ctx.strokeStyle = col("--line");
    ctx.lineWidth = 1;
    [0, 0.25, 0.5, 0.75, 1].forEach((v) => {
      const y = T + ih * (1 - v);
      ctx.beginPath();
      ctx.moveTo(L, y);
      ctx.lineTo(width - R, y);
      ctx.stroke();
      ctx.fillText(v.toFixed(2), 4, y + 4);
    });
    const N = 160;
    const ev = events.slice(-N);
    const step = iw / (N - 1);
    const x0 = L + iw - (ev.length - 1) * step;
    // ground-truth shading
    ctx.fillStyle = col("--malicious") + "22";
    ev.forEach((e, i) => {
      if (e.label) ctx.fillRect(x0 + i * step - step / 2, T, step, ih);
    });
    // threshold
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = col("--muted");
    ctx.beginPath();
    ctx.moveTo(L, T + ih / 2);
    ctx.lineTo(width - R, T + ih / 2);
    ctx.stroke();
    ctx.setLineDash([]);
    // sentinel dots
    ev.forEach((e, i) => {
      const o = e.outputs[primary];
      if (!o || o.sentinel_score == null) return;
      ctx.fillStyle = o.exit ? col("--sentinel") : col("--analyst");
      ctx.beginPath();
      ctx.arc(x0 + i * step, T + ih * (1 - o.sentinel_score), 2.6, 0, Math.PI * 2);
      ctx.fill();
    });
    // final score line
    ctx.strokeStyle = col("--text");
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    ev.forEach((e, i) => {
      const o = e.outputs[primary];
      if (!o) return;
      const x = x0 + i * step, y = T + ih * (1 - o.score);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    if (ev.length) {
      const o = ev[ev.length - 1].outputs[primary];
      if (o) {
        ctx.fillStyle = o.pred > 0 ? col("--malicious") : col("--benign");
        ctx.beginPath();
        ctx.arc(L + iw, T + ih * (1 - o.score), 5, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    ctx.fillStyle = col("--muted");
    ctx.fillText(`last ${ev.length} windows →`, L, height - 6);
  }, [events, primary, width, height]);

  return (
    <div ref={wrap} style={{ width: "100%" }}>
      <canvas ref={canvas} style={{ width: "100%", height, display: "block" }} />
    </div>
  );
}
