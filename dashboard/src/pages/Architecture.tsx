import { motion } from "framer-motion";
import { useEffect, useState } from "react";
import { Tile } from "../components/ui";
import { api, fmt } from "../lib/api";
import { useLive } from "../lib/live";

type Arch = {
  config: Record<string, unknown>;
  gate: Record<string, number | boolean>;
  params: Record<string, number> | null;
  macs: { sentinel_path: number | null; full: number | null; expected: number | null };
};

const BLOCKS = [
  { id: "pip", label: "PIP layer", detail: "Linear F→d, initialised with the top-d PCA axes of the train split: z = W(x − μ)", color: "var(--accent)" },
  { id: "stem", label: "DS stem", detail: "depthwise k=3 + pointwise d→c, BatchNorm, ReLU6", color: "var(--accent)" },
  { id: "sentinel_exit_only", label: "Sentinel exit", detail: "one more DS block, global average pooling, linear → p₁ = softmax(z/T₁)", color: "var(--sentinel)" },
  { id: "analyst_exit_only", label: "Analyst exit", detail: "2 residual inverted-bottleneck DS-TCN blocks (dilation 1, 2) + BiGRU (R only) + attention pooling", color: "var(--analyst)" },
];

const EQUATIONS = [
  ["(1) PIP", "z_t = W_p (x_t − μ),  W_p ∈ ℝ^{d×F}"],
  ["(2) Sentinel", "p₁ = softmax(W₁ · GAP(H) / T₁)"],
  ["(3) Gate", "exit ⇔ (ŷ₁ = 0 ∧ p₁,₀ ≥ τ_b) ∨ (ŷ₁ ≠ 0 ∧ max p₁ ≥ τ_m)"],
  ["(4) Risk control", "τ_b = min{ τ : U_CP(m(τ), N_m; α) ≤ ε }"],
  ["(5) Training loss", "L = CE(p₁,y) + CE(p₂,y) + γ·τ²·KL(σ(t/τ) ‖ σ(s₂/τ)) + δ·KL(p₂ ‖ p₁)"],
  ["(6) Expected cost", "E[C] = C₁ + (1 − ρ)·C₂,   S = (C₁ + C₂) / E[C]"],
];

export function ArchitecturePage() {
  const { cards } = useLive();
  const [arch, setArch] = useState<Arch | null>(null);
  useEffect(() => {
    api.get<Arch>("/api/architecture").then(setArch).catch(() => undefined);
  }, []);
  const p = arch?.params;
  const total = p?.total ?? 1;
  const cfg = arch?.config as Record<string, number | string | number[]> | undefined;

  return (
    <>
      <div className="page-head">
        <div>
          <h2>Architecture</h2>
          <p>LiteCascade as deployed, read from the model bundle, followed by every deep-learning model in the comparison.</p>
        </div>
      </div>
      <div className="tiles">
        <Tile k="Total parameters" v={fmt.int(p?.total)} d={`F=${cfg?.F_in} · d=${cfg?.d} · c=${cfg?.c} · variant ${cfg?.variant}`} />
        <Tile k="Sentinel path MACs" v={fmt.si(arch?.macs.sentinel_path)} d="PIP + stem + Sentinel (C₁)" accent="var(--sentinel)" />
        <Tile k="Full path MACs" v={fmt.si(arch?.macs.full)} d="C₁ + C₂ (both exits)" accent="var(--analyst)" />
        <Tile k="Expected MACs" v={fmt.si(arch?.macs.expected)} d="E[C] at the measured exit rate" accent="var(--ok)" />
      </div>

      <div className="grid g2">
        <div className="panel">
          <div className="panel-head"><h3>Parameter budget by block</h3></div>
          <div style={{ display: "flex", height: 26, borderRadius: 8, overflow: "hidden", gap: 2, marginBottom: 16 }}>
            {BLOCKS.map((b, i) => (
              <motion.div key={b.id} title={`${b.label}: ${fmt.int(p?.[b.id])}`}
                initial={{ width: 0 }} animate={{ width: `${((p?.[b.id] ?? 0) / total) * 100}%` }}
                transition={{ delay: i * 0.15, type: "spring", stiffness: 60 }} style={{ background: b.color, opacity: 0.4 + i * 0.15 }} />
            ))}
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {BLOCKS.map((b, i) => (
              <motion.div key={b.id} initial={{ opacity: 0, x: -14 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.1 * i }}
                style={{ display: "grid", gridTemplateColumns: "10px 1fr auto", gap: 10, alignItems: "baseline" }}>
                <span style={{ width: 10, height: 10, borderRadius: 3, background: b.color, opacity: 0.4 + i * 0.15 }} />
                <div><b>{b.label}</b><div className="hint">{b.detail}</div></div>
                <span className="num">{fmt.int(p?.[b.id])}</span>
              </motion.div>
            ))}
          </div>
          <p className="hint" style={{ marginTop: 12 }}>
            Paper design targets for F = 115: Sentinel exit &lt; 2K parameters, ≈ 20K total, ≈ 0.06M / 0.18M MACs for the Sentinel / full path.
          </p>
        </div>
        <div className="panel">
          <div className="panel-head"><h3>Equations implemented</h3></div>
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {EQUATIONS.map(([k, v], i) => (
              <motion.div key={k} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.08 * i }}
                style={{ padding: "8px 12px", background: "var(--panel-2)", borderRadius: 10, border: "1px solid var(--line)" }}>
                <div className="hint">{k}</div>
                <div className="mono" style={{ fontSize: 13 }}>{v}</div>
              </motion.div>
            ))}
          </div>
          <p className="hint" style={{ marginTop: 10 }}>
            Deployed gate: T₁ = {fmt.num(arch?.gate.T1 as number)}, T₂ = {fmt.num(arch?.gate.T2 as number)}, τ_b = {fmt.num(arch?.gate.tau_b as number, 4)},
            τ_m = {fmt.num(arch?.gate.tau_m as number, 4)}, ε = {String(arch?.gate.epsilon)}, α = {String(arch?.gate.alpha)}.
          </p>
        </div>
      </div>

      <div className="panel" style={{ marginTop: 16 }}>
        <div className="panel-head"><h3>Deep-learning model zoo</h3><span className="hint">{cards.length} models, all trained on the same splits</span></div>
        <div className="arena">
          {cards.map((c, i) => (
            <motion.div key={c.id} className="mcard" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.03 * i }}
              whileHover={{ y: -3 }}>
              <div className="name">{c.name}</div>
              <div className="fam">{c.family}{c.quantization !== "none" ? ` · ${c.quantization} INT8` : ""}</div>
              <p className="hint" style={{ margin: "8px 0" }}>{c.description}</p>
              <div className="num" style={{ fontSize: 12, display: "flex", justifyContent: "space-between" }}>
                <span>{fmt.si(c.metrics.params)} params</span><span>F1 {fmt.num(c.metrics.f1_macro)}</span>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </>
  );
}
