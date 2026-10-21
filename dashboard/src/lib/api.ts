export type Metrics = {
  f1_macro: number | null;
  f1_ci?: [number, number] | null;
  recall_malicious: number | null;
  fpr: number | null;
  mcc?: number | null;
  params: number | null;
  macs: number | null;
  macs_full?: number | null;
  macs_sentinel?: number | null;
  size_kb: number | null;
  latency_p50_ms: number | null;
  latency_sentinel_ms?: number | null;
  rho?: number | null;
  exit_induced_miss_rate?: number | null;
  sentinel_f1?: number | null;
  analyst_f1?: number | null;
  measured_speedup?: number | null;
  param_breakdown?: Record<string, number> | null;
};

export type ModelCard = {
  id: string;
  name: string;
  kind: "cascade" | "baseline";
  arch: string;
  family: string;
  description: string;
  quantization: string;
  dataset: string;
  is_synthetic: boolean;
  window: number;
  n_features: number;
  class_names: string[];
  gate: Record<string, number | boolean | null> | null;
  metrics: Metrics;
};

export type ModelOutput = {
  pred: number;
  score: number;
  latency_ms: number;
  exit?: boolean;
  sentinel_score?: number;
  sentinel_benign?: number;
};

export type LiveEvent = {
  step: number;
  window_index: number;
  segment: string;
  family: string;
  label: number;
  primary: string;
  outputs: Record<string, ModelOutput>;
  ts: number;
};

export type ModelLive = {
  n: number;
  accuracy: number;
  f1_macro: number;
  recall: number;
  fpr: number;
  tp: number;
  fp: number;
  fn: number;
  tn: number;
  latency_ms_recent: number | null;
  latency_ms_mean: number | null;
  latency_spark: number[];
};

export type LiveSummary = {
  steps: number;
  primary: string;
  exit_rate: number | null;
  analyst_calls: number;
  compute_saved: number | null;
  alerts: number;
  models: Record<string, ModelLive>;
  queue: number;
  families: string[];
};

export type LivePayload = { event: LiveEvent | null; summary: LiveSummary; running: boolean; speed: number };

export type GatePoint = {
  epsilon: number;
  alpha: number;
  tau_b: number;
  feasible: boolean;
  N_m: number;
  min_N_m: number;
  m_max: number;
  rho: number;
  exit_induced_miss_rate: number;
  f1_macro: number;
  recall: number;
  fpr: number;
};

async function handle(r: Response) {
  if (!r.ok) {
    let detail = r.statusText;
    try {
      detail = (await r.json()).detail ?? detail;
    } catch {
      /* not JSON */
    }
    throw new Error(detail);
  }
  const ct = r.headers.get("content-type") ?? "";
  return ct.includes("json") ? r.json() : r.text();
}

export const api = {
  get: <T,>(path: string): Promise<T> => fetch(path).then(handle),
  post: <T,>(path: string, body: unknown): Promise<T> =>
    fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).then(handle),
  form: <T,>(path: string, fd: FormData): Promise<T> => fetch(path, { method: "POST", body: fd }).then(handle),
};

export const fmt = {
  pct: (v: number | null | undefined, d = 1) => (v == null ? "–" : `${(v * 100).toFixed(d)}%`),
  num: (v: number | null | undefined, d = 3) => (v == null ? "–" : v.toFixed(d)),
  int: (v: number | null | undefined) => (v == null ? "–" : Math.round(v).toLocaleString()),
  si: (v: number | null | undefined) => {
    if (v == null) return "–";
    if (v >= 1e6) return `${(v / 1e6).toFixed(2)}M`;
    if (v >= 1e3) return `${(v / 1e3).toFixed(1)}K`;
    return `${Math.round(v)}`;
  },
  ms: (v: number | null | undefined) => (v == null ? "–" : v < 1 ? `${(v * 1000).toFixed(0)} µs` : `${v.toFixed(2)} ms`),
  family: (f: string) => f.replace(/_/g, " "),
};
