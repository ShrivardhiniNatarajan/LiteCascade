import React from "react";
import { Layers, GitMerge } from "lucide-react";
import { useStore } from "../store/useStore";
import { SourceTooltip } from "./SourceTooltip";

export const PipStemCard: React.FC = () => {
  const { meta } = useStore();

  const d = meta?.d.value ?? 16;
  const c = meta?.c.value ?? 24;

  return (
    <div className="w-full bg-[#0E1420] border border-[#1B2536] rounded-card p-5 relative overflow-hidden transition-all shadow-sm">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Layers className="w-4 h-4 text-[#34D399]" />
          <h2 className="text-xs font-mono font-bold tracking-wider uppercase text-[#E6EBF5]">
            2. PIP Layer + Shared Stem
          </h2>
          <SourceTooltip source={meta?.d.source} />
        </div>
        <div className="text-xs font-mono font-bold text-[#34D399] bg-[#060A12] border border-[#1B2536] px-2.5 py-0.5 rounded-full">
          d={d} → c={c}
        </div>
      </div>

      {/* Architecture specifications */}
      <div className="bg-[#060A12] border border-[#1B2536] rounded-xl p-3 flex flex-col gap-2 font-mono text-xs">
        <div className="flex items-center justify-between">
          <span className="text-[#E6EBF5] font-medium flex items-center gap-1.5">
            <GitMerge className="w-3.5 h-3.5 text-[#34D399]" />
            <span>DS-Conv3 (MobileNetV2-style)</span>
          </span>
          <span className="text-[10px] text-[#7C8AA3]">Depthwise-Separable</span>
        </div>

        <div className="flex items-center justify-between text-[11px] text-[#7C8AA3] pt-1 border-t border-[#1B2536]">
          <span>Linear Projection: Initialised via Incremental PCA</span>
          <span className="text-[#34D399]">1D Temporal Stem</span>
        </div>
      </div>
    </div>
  );
};
