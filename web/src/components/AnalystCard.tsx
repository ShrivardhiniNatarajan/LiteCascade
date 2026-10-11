import React from "react";
import { Cpu, Activity, ShieldCheck, AlertOctagon } from "lucide-react";
import { useStore } from "../store/useStore";
import { SourceTooltip } from "./SourceTooltip";

export const AnalystCard: React.FC = () => {
  const { meta, lastEvent } = useStore();

  const isAnalystActive = lastEvent ? lastEvent.exit === "analyst" : false;
  const isVariantR = meta?.variant.value?.toUpperCase().includes("R") ?? false;
  const analystParams = meta?.params.value.analyst ?? 68432;
  const analystMacs = meta?.macs.value.analyst ?? 0.221;

  const analystVerdict = lastEvent?.analyst_verdict ?? (lastEvent?.y_pred === 1 ? "Malicious" : "Benign");

  return (
    <div
      className={`w-full bg-[#0E1420] border rounded-card p-5 relative overflow-hidden transition-all duration-300 ${
        isAnalystActive
          ? "border-[#C084FC] shadow-glowViolet ring-1 ring-[#C084FC]/40"
          : "border-[#1B2536] shadow-sm"
      }`}
    >
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
        <div className="flex items-center gap-2">
          <Cpu className={`w-4 h-4 ${isAnalystActive ? "text-[#C084FC]" : "text-[#7C8AA3]"}`} />
          <h2 className="text-xs font-mono font-bold tracking-wider uppercase text-[#E6EBF5]">
            4. Exit 2 • Analyst
          </h2>
          <SourceTooltip source={meta?.params.source} />
        </div>
        <div className="flex items-center gap-2 text-xs font-mono">
          <span className="text-[#C084FC] bg-[#060A12] border border-[#1B2536] px-2 py-0.5 rounded-full text-[10px]">
            {analystParams.toLocaleString()} params
          </span>
          <span className="text-[#7C8AA3] text-[10px] bg-[#060A12] border border-[#1B2536] px-2 py-0.5 rounded-full">
            {analystMacs} M MACs
          </span>
        </div>
      </div>

      {/* Main Body */}
      <div className="bg-[#060A12] border border-[#1B2536] rounded-xl p-3 flex flex-col gap-2.5 font-mono text-xs">
        <div className="flex items-center justify-between">
          <span className="text-[#E6EBF5] font-semibold text-xs flex items-center gap-1.5">
            <Activity className="w-3.5 h-3.5 text-[#C084FC]" />
            <span>Deep Temporal Analysis</span>
          </span>
          <span
            className={`px-2 py-0.5 rounded-full text-[10px] uppercase font-bold tracking-wider border ${
              isAnalystActive
                ? "bg-[#C084FC]/15 text-[#C084FC] border-[#C084FC]/40 animate-pulse"
                : "bg-[#1B2536]/40 text-[#7C8AA3] border-[#1B2536]"
            }`}
          >
            {isAnalystActive ? "Status: Active" : "Status: Idle"}
          </span>
        </div>

        <div className="flex items-center justify-between text-[11px] text-[#7C8AA3]">
          <span>
            DS-TCN (Dilated 1,2) + Attention
            {isVariantR && <strong className="text-[#C084FC]"> + BiGRU</strong>}
          </span>
          <span className="text-[10px] text-[#7C8AA3]">
            {isVariantR ? "Variant R (Recurrent)" : "Standard Non-recurrent"}
          </span>
        </div>

        {/* Verdict Strip */}
        <div
          className={`mt-1 py-2 px-3 rounded-lg text-xs font-bold tracking-wide flex items-center justify-between transition-all ${
            !isAnalystActive
              ? "bg-[#1B2536]/30 text-[#7C8AA3] border border-[#1B2536]"
              : analystVerdict === "Benign"
              ? "bg-[#C084FC]/10 text-[#C084FC] border border-[#C084FC]/30"
              : "bg-[#F43F5E]/10 text-[#F43F5E] border border-[#F43F5E]/30"
          }`}
        >
          <span className="flex items-center gap-1.5">
            {!isAnalystActive ? (
              <ShieldCheck className="w-3.5 h-3.5 text-[#7C8AA3]" />
            ) : analystVerdict === "Benign" ? (
              <ShieldCheck className="w-3.5 h-3.5 text-[#C084FC]" />
            ) : (
              <AlertOctagon className="w-3.5 h-3.5 text-[#F43F5E]" />
            )}
            <span>
              {!isAnalystActive
                ? "Analyst Bypassed (0 MACs)"
                : `Analyst verdict: ${analystVerdict}`}
            </span>
          </span>
          {isAnalystActive && (
            <span className="text-[10px] uppercase font-mono text-[#7C8AA3]">
              Cost: {analystMacs} M MACs
            </span>
          )}
        </div>
      </div>
    </div>
  );
};
