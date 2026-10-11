/**
 * Typed client for LiteCascade FastAPI backend (Stage 16C).
 * Based on OpenAPI schema from /api.
 */

export interface SourcedValue<T = any> {
  value: T;
  source: string;
}

export interface MetaResponse {
  data_source: SourcedValue<string>;
  is_synthetic: SourcedValue<boolean>;
  run_id: SourcedValue<string>;
  git_commit: SourcedValue<string>;
  model_file: SourcedValue<string>;
  model_format: SourcedValue<"TFLite INT8" | "PyTorch FP32">;
  gate_loaded: SourcedValue<boolean>;
  tau_b: SourcedValue<number>;
  tau_m: SourcedValue<number>;
  epsilon: SourcedValue<number>;
  variant: SourcedValue<string>;
  params: SourcedValue<{ sentinel: number; analyst: number; total: number }>;
  macs: SourcedValue<{ sentinel: number; analyst: number; total: number }>;
  d: SourcedValue<number>;
  c: SourcedValue<number>;
  T_len: SourcedValue<number>;
  F_in: SourcedValue<number>;
  available_families: SourcedValue<string[]>;
}

export interface StatsResponse {
  n: SourcedValue<number>;
  exit_rate: SourcedValue<number>;
  compute_saved_vs_always_analyst: SourcedValue<number>;
  latency_p50: SourcedValue<number>;
  latency_p95: SourcedValue<number>;
  accuracy: SourcedValue<number | null>;
  accuracy_ci_low: SourcedValue<number | null>;
  accuracy_ci_high: SourcedValue<number | null>;
  recall_loss: SourcedValue<number>;
  int8_arena_ram_kb: SourcedValue<number>;
  gate_loaded: SourcedValue<boolean>;
}

export interface StreamEvent {
  idx: number;
  flow_id: string;
  y_true: number;
  y_pred: number;
  family_pred: string;
  confidence: number;
  tau_b_used: number;
  exit: "sentinel" | "analyst";
  sentinel_verdict: "Benign" | "Malicious" | "Uncertain";
  analyst_verdict: "Benign" | "Malicious" | null;
  latency_ms: number;
  macs_used: number;
  is_injected: boolean;
  timestamp: string;
}

export interface StreamControlResponse {
  action: string;
  is_running: boolean;
  speed: number;
  total_windows: number;
  current_index: number;
}

export interface InjectRequest {
  family: string;
  duration_windows: number;
}

export interface InjectResponse {
  status: string;
  family: string;
  start_index: number;
  duration_windows: number;
}

const API_BASE = "";

export async function fetchMeta(): Promise<MetaResponse> {
  const res = await fetch(`${API_BASE}/api/meta`);
  if (!res.ok) {
    throw new Error(`Failed to fetch metadata: ${res.statusText}`);
  }
  return res.json();
}

export async function fetchStats(): Promise<StatsResponse> {
  const res = await fetch(`${API_BASE}/api/stats`);
  if (!res.ok) {
    throw new Error(`Failed to fetch stats: ${res.statusText}`);
  }
  return res.json();
}

export async function fetchDevice(): Promise<Record<string, any>> {
  const res = await fetch(`${API_BASE}/api/device`);
  if (!res.ok) {
    throw new Error(`Failed to fetch device benchmarks: ${res.statusText}`);
  }
  return res.json();
}

export async function controlStream(
  action: "start" | "pause" | "reset" | "step",
  speed?: number
): Promise<StreamControlResponse> {
  const res = await fetch(`${API_BASE}/api/stream/${action}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ speed }),
  });
  if (!res.ok) {
    throw new Error(`Stream action ${action} failed: ${res.statusText}`);
  }
  return res.json();
}

export async function injectAttack(payload: InjectRequest): Promise<InjectResponse> {
  const res = await fetch(`${API_BASE}/api/inject`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const errorBody = await res.json().catch(() => ({}));
    throw new Error(errorBody.detail || `Attack injection failed: ${res.statusText}`);
  }
  return res.json();
}
