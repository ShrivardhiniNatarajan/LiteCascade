import React from "react";
import { Zap, CheckCircle2, AlertTriangle, ArrowRight } from "lucide-react";
import { useStore } from "../store/useStore";
import { SourceTooltip } from "./SourceTooltip";

export const SentinelCard: React.FC = () => {
  const { meta, lastEvent } = useStore();

  const isSentinelExit = lastEvent ? lastEvent.exit === "sentinel" : false;
  const confidence = lastEvent ? lastEvent.confidence : 0.95;
  const confPct = Math.round(confidence * 100);

  const tau_b = meta?.tau_b.value ?? 0.90;
  const tau_b_pct = Math.round(tau_b * 100);

  const sentinelVerdict = lastEvent?.sentinel_verdict ?? "Benign";

  return (
    <div
      className={`w-full bg-[#0E1420] border rounded-card p-5 relative overflow-hidden transition-all duration-300 ${
        isSentinelExit
          ? "border-[#34D399] shadow-glowGreen ring-1 ring-[#34D399]/40"
          : "border-[#1B2536] shadow-sm"
      }`}
    >
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
        <div className="flex items-center gap-2">
          <Zap className={`w-4 h-4 ${isSentinelExit ? "text-[#34D399]" : "text-[#7C8AA3]"}`} />
          <h2 className="text-xs font-mono font-bold tracking-wider uppercase text-[#E6EBF5]">
            3. Exit 1 • Sentinel
          </h2>
          <SourceTooltip source={meta?.params.source} />
        </div>
        <div className="flex items-center gap-2 text-xs font-mono">
          <span className="text-[#34D399] bg-[#060A12] border border-[#1B2536] px-2 py-0.5 rounded-full text-[10px]">
            {meta?.params.value.sentinel ?? 2410} params
          </span>
          <span className="text-[#7C8AA3] text-[10px] bg-[#060A12] border border-[#1B2536] px-2 py-0.5 rounded-full">
            {meta?.macs.value.sentinel ?? 0.008} M MACs
          </span>
        </div>
      </div>

      {/* Confidence Gate Bar with tau_b Marker */}
      <div className="bg-[#060A12] border border-[#1B2536] rounded-xl p-3 flex flex-col gap-2.5 font-mono text-xs">
        <div className="flex items-center justify-between">
          <span className="text-[#7C8AA3] flex items-center gap-1">
            <span>Confidence Gate:</span>
            <span className="text-[#E6EBF5] font-bold text-sm">{confPct}%</span>
          </span>
          <span className="text-[#7C8AA3] text-[11px]">
            Threshold (τ_b): <strong className="text-[#38BDF8]">{tau_b.toFixed(2)}</strong>
          </span>
        </div>

        {/* Progress Track */}
        <div className="relative w-full h-3 bg-[#1B2536] rounded-full overflow-hidden">
          {/* Animated Fill Bar */}
          <div
            className={`h-full transition-all duration-300 rounded-full ${
              confPct >= tau_b_pct ? "bg-[#34D399]" : "bg-[#FBBF24]"
            }`}
            style={{ width: `${Math.min(100, Math.max(5, confPct))}%` }}
          />
          {/* Marker for tau_b */}
          <div
            className="absolute top-0 bottom-0 w-0.5 bg-white shadow z-10"
            style={{ left: `${tau_b_pct}%` }}
            title={`Threshold τ_b = ${tau_b}`}
          />
        </div>

        {/* Verdict Strip */}
        <div
          className={`mt-1 py-2 px-3 rounded-lg text-xs font-bold tracking-wide flex items-center justify-between transition-all ${
            sentinelVerdict === "Benign"
              ? "bg-[#34D399]/10 text-[#34D399] border border-[#34D399]/30"
              : sentinelVerdict === "Malicious"
              ? "bg-[#F43F5E]/10 text-[#F43F5E] border border-[#F43F5E]/30"
              : "bg-[#FBBF24]/10 text-[#FBBF24] border border-[#FBBF24]/30"
          }`}
        >
          <span className="flex items-center gap-1.5">
            {sentinelVerdict === "Benign" ? (
              <CheckCircle2 className="w-3.5 h-3.5" />
            ) : sentinelVerdict === "Malicious" ? (
              <Zap className="w-3.5 h-3.5" />
            ) : (
              <AlertTriangle className="w-3.5 h-3.5" />
            )}
            <span>
              {sentinelVerdict === "Benign"
                ? "EXIT 1 GRANTED (Benign)"
                : sentinelVerdict === "Malicious"
                ? "EXIT 1 GRANTED (Malicious)"
                : "UNCERTAIN → ESCALATED"}
            </span>
          </span>
          {sentinelVerdict === "Uncertain" && (
            <span className="text-[10px] uppercase font-mono flex items-center gap-1">
              Escalate <ArrowRight className="w-3 h-3" />
            </span>
          )}
        </div>
      </div>
    </div>
  );
};
