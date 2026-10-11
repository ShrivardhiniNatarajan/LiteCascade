import React from "react";
import { Radio } from "lucide-react";
import { useStore } from "../store/useStore";
import { SourceTooltip } from "./SourceTooltip";

export const InputIngestCard: React.FC = () => {
  const { meta, lastEvent } = useStore();

  const T_len = meta?.T_len.value ?? 10;
  const F_in = meta?.F_in.value ?? 115;
  const dataLabel = meta?.is_synthetic.value ? "Synthetic fixture" : "N-BaIoT test split";

  const flowId = lastEvent?.flow_id ?? "flow_0";
  const isInjected = lastEvent?.is_injected ?? false;
  const familyName = lastEvent?.family_pred ?? "benign";
  const isMalicious = lastEvent?.y_true === 1;

  return (
    <div className="w-full bg-[#0E1420] border border-[#1B2536] rounded-card p-5 relative overflow-hidden transition-all shadow-sm">
      {/* Card Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Radio className="w-4 h-4 text-[#38BDF8]" />
          <h2 className="text-xs font-mono font-bold tracking-wider uppercase text-[#E6EBF5]">
            1. Input Ingest
          </h2>
          <SourceTooltip source={meta?.data_source.source} />
        </div>
        <div className="text-xs font-mono font-bold text-[#38BDF8] bg-[#060A12] border border-[#1B2536] px-2.5 py-0.5 rounded-full">
          T={T_len}, F={F_in}
        </div>
      </div>

      {/* Main Flow Indicator Line */}
      <div className="bg-[#060A12] border border-[#1B2536] rounded-xl p-3 flex flex-col gap-1.5 font-mono">
        <div className="flex items-center justify-between text-xs">
          <span className="font-bold text-[#E6EBF5] flex items-center gap-1.5">
            <span className="text-[#7C8AA3]">{flowId}</span>
            <span>•</span>
            <span className={isInjected ? "text-[#F43F5E] font-bold" : isMalicious ? "text-[#FBBF24]" : "text-[#34D399]"}>
              {isInjected
                ? `Injected ${familyName.toUpperCase()} Flow`
                : isMalicious
                ? `${familyName.toUpperCase()} Attack Window`
                : "Standard Benign Flow"}
            </span>
          </span>
          <span className="text-[10px] text-[#7C8AA3]">
            idx #{lastEvent?.idx ?? 0}
          </span>
        </div>

        {/* Provenance Subline */}
        <div className="text-[11px] text-[#7C8AA3] flex items-center justify-between">
          <span>Source: {dataLabel}</span>
          <span className="text-[10px] text-[#38BDF8]">Sliding window (stride=1)</span>
        </div>
      </div>
    </div>
  );
};
