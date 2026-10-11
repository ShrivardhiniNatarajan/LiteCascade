import React from "react";
import { Gauge, HelpCircle, CheckCircle } from "lucide-react";
import { useStore } from "../store/useStore";
import { SourceTooltip } from "./SourceTooltip";

export const EdgeTelemetryCard: React.FC = () => {
  const { stats, device } = useStore();

  const n = stats?.n.value ?? 0;
  const exitRatePct = stats ? (stats.exit_rate.value * 100).toFixed(1) : "0.0";
  const p50 = stats ? stats.latency_p50.value.toFixed(2) : "0.00";
  const p95 = stats ? stats.latency_p95.value.toFixed(2) : "0.00";

  // Check hardware provenance for latency
  const hasMeasuredRpi =
    device &&
    Object.values(device).some(
      (d: any) =>
        d?.measured_or_emulated === "measured" &&
        (d?.device_name?.includes("Raspberry Pi 4B") || d?.hardware?.includes("Raspberry Pi 4B"))
    );
  const latencyHost = hasMeasuredRpi ? "Raspberry Pi 4B" : "this host, CPU (Emulated)";

  // INT8 RAM
  const arenaRam = stats?.int8_arena_ram_kb.value ?? 20.0;
  const ramSource = stats?.int8_arena_ram_kb.source ?? "estimated";
  const isRamMeasured = ramSource.toLowerCase().includes("measured");

  // Compute saved vs always analyst
  const computeSaved = stats ? stats.compute_saved_vs_always_analyst.value * 100 : 0;

  // Accuracy display (strictly - until n >= 50)
  const accuracyVal = stats?.accuracy.value;
  const accLow = stats?.accuracy_ci_low.value;
  const accHigh = stats?.accuracy_ci_high.value;
  const showAccuracy = n >= 50 && accuracyVal !== null && accuracyVal !== undefined;

  return (
    <div className="w-full bg-[#0E1420] border border-[#1B2536] rounded-card p-5 relative overflow-hidden transition-all shadow-sm flex flex-col gap-4">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-[#1B2536]">
        <div className="flex items-center gap-2">
          <Gauge className="w-4 h-4 text-[#34D399]" />
          <h2 className="text-xs font-mono font-bold tracking-wider uppercase text-[#E6EBF5]">
            6. Edge Performance Telemetry
          </h2>
          <SourceTooltip source={stats?.exit_rate.source} />
        </div>
        <div className="text-[10px] font-mono text-[#7C8AA3]">
          {showAccuracy ? (
            <span className="text-[#34D399] bg-[#060A12] border border-[#1B2536] px-2 py-0.5 rounded">
              Accuracy (n={n}, 95% CI): {(accuracyVal! * 100).toFixed(1)}% [
              {(accLow! * 100).toFixed(1)}%, {(accHigh! * 100).toFixed(1)}%]
            </span>
          ) : (
            <span className="bg-[#060A12] border border-[#1B2536] px-2 py-0.5 rounded">
              Accuracy (n={n}, 95% CI): <strong className="text-[#7C8AA3]">-</strong>
            </span>
          )}
        </div>
      </div>

      {/* 2x2 Tiles Grid */}
      <div className="grid grid-cols-2 gap-3 font-mono">
        {/* Tile 1: Early Exit Rate */}
        <div className="bg-[#060A12] border border-[#1B2536] rounded-xl p-3 flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <span className="text-[11px] text-[#7C8AA3] uppercase tracking-wider font-semibold">
              Early Exit Rate
            </span>
            <SourceTooltip source={stats?.exit_rate.source} />
          </div>
          <div className="my-1.5 flex items-baseline gap-1">
            <span className="text-2xl font-bold text-[#34D399] tracking-tight">
              {exitRatePct}%
            </span>
          </div>
          <div className="text-[10px] text-[#7C8AA3] flex items-center justify-between">
            <span>H3 target: ≥70%</span>
            {parseFloat(exitRatePct) >= 70 && (
              <CheckCircle className="w-3 h-3 text-[#34D399]" />
            )}
          </div>
        </div>

        {/* Tile 2: Avg Latency */}
        <div className="bg-[#060A12] border border-[#1B2536] rounded-xl p-3 flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <span className="text-[11px] text-[#7C8AA3] uppercase tracking-wider font-semibold">
              Avg Latency (p50)
            </span>
            <SourceTooltip source={stats?.latency_p50.source} />
          </div>
          <div className="my-1.5 flex items-baseline gap-1">
            <span className="text-2xl font-bold text-[#38BDF8] tracking-tight">
              {p50}
            </span>
            <span className="text-xs text-[#7C8AA3]">ms</span>
            <span className="text-[10px] text-[#7C8AA3] ml-auto">p95: {p95}ms</span>
          </div>
          <div className="text-[10px] text-[#7C8AA3] truncate" title={latencyHost}>
            {latencyHost}
          </div>
        </div>

        {/* Tile 3: INT8 Arena RAM */}
        <div className="bg-[#060A12] border border-[#1B2536] rounded-xl p-3 flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <span className="text-[11px] text-[#7C8AA3] uppercase tracking-wider font-semibold">
              INT8 Arena RAM
            </span>
            <span
              className={`text-[9px] px-1.5 py-0.2 rounded font-bold uppercase tracking-wider border ${
                isRamMeasured
                  ? "bg-[#34D399]/15 text-[#34D399] border-[#34D399]/40"
                  : "bg-[#FBBF24]/15 text-[#FBBF24] border-[#FBBF24]/40"
              }`}
            >
              {isRamMeasured ? "MEASURED" : "ESTIMATED"}
            </span>
          </div>
          <div className="my-1.5 flex items-baseline gap-1">
            <span className="text-2xl font-bold text-[#C084FC] tracking-tight">
              {arenaRam.toFixed(1)}
            </span>
            <span className="text-xs text-[#7C8AA3]">KB</span>
          </div>
          <div className="text-[10px] text-[#7C8AA3]">
            Budget ≤ 32 KB (MCU)
          </div>
        </div>

        {/* Tile 4: Total Samples */}
        <div className="bg-[#060A12] border border-[#1B2536] rounded-xl p-3 flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <span className="text-[11px] text-[#7C8AA3] uppercase tracking-wider font-semibold">
              Total Samples
            </span>
            <SourceTooltip source={stats?.n.source} />
          </div>
          <div className="my-1.5 flex items-baseline gap-1">
            <span className="text-2xl font-bold text-[#E6EBF5] tracking-tight">
              {n.toLocaleString()}
            </span>
            <span className="text-xs text-[#7C8AA3]">windows</span>
          </div>
          <div className="text-[10px] text-[#7C8AA3]">
            Live Replay Engine
          </div>
        </div>
      </div>

      {/* Compute Saved Bar */}
      <div className="bg-[#060A12] border border-[#1B2536] rounded-xl p-3 flex flex-col gap-2 font-mono">
        <div className="flex items-center justify-between text-xs">
          <div className="flex items-center gap-1.5">
            <span className="text-[#E6EBF5] font-semibold">Compute Saved (MACs)</span>
            <div
              className="text-[#7C8AA3] cursor-help"
              title="Formula: 1 - E[C] / C_always_analyst (Savings vs executing full DS-TCN Analyst on every window)"
            >
              <HelpCircle className="w-3.5 h-3.5" />
            </div>
            <SourceTooltip source={stats?.compute_saved_vs_always_analyst.source} />
          </div>
          <span className="text-[#34D399] font-bold text-sm">
            {computeSaved.toFixed(1)}%
          </span>
        </div>

        <div className="w-full h-2.5 bg-[#1B2536] rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-[#34D399] to-[#38BDF8] rounded-full transition-all duration-300"
            style={{ width: `${Math.min(100, Math.max(0, computeSaved))}%` }}
          />
        </div>

        <div className="text-[10px] text-[#7C8AA3] flex items-center justify-between">
          <span>Based on live Sentinel vs Analyst inference counters</span>
          <span>1 - E[C]/C_always_analyst</span>
        </div>
      </div>
    </div>
  );
};
