import React from "react";
import { ShieldAlert, AlertCircle } from "lucide-react";
import { InlineMath } from "react-katex";
import { useStore } from "../store/useStore";
import { SourceTooltip } from "./SourceTooltip";

export const SafetyNetCard: React.FC = () => {
  const { meta, stats } = useStore();

  const tau_b = meta?.tau_b.value ?? 0.90;
  const epsilon = meta?.epsilon.value ?? 0.05;
  const recallLoss = stats?.recall_loss.value ?? 0.0;
  const recallLossPct = (recallLoss * 100).toFixed(2);
  const isGateLoaded = meta?.gate_loaded.value ?? false;

  return (
    <div className="w-full bg-[#0E1420] border border-[#1B2536] rounded-card p-5 relative overflow-hidden transition-all shadow-sm flex flex-col gap-3">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-2 pb-2 border-b border-[#1B2536]">
        <div className="flex items-center gap-2">
          <ShieldAlert className="w-4 h-4 text-[#FBBF24]" />
          <h2 className="text-xs font-mono font-bold tracking-wider uppercase text-[#FBBF24]">
            7. Statistical Safety Net
          </h2>
          <SourceTooltip source={meta?.epsilon.source} />
        </div>
        <span
          className={`text-[10px] font-mono px-2 py-0.5 rounded font-semibold border ${
            isGateLoaded
              ? "bg-[#FBBF24]/10 border-[#FBBF24]/30 text-[#FBBF24]"
              : "bg-[#F43F5E]/15 border-[#F43F5E]/40 text-[#F43F5E]"
          }`}
        >
          {isGateLoaded ? "Risk-Controlled Exit Bound" : "Gate Not Loaded"}
        </span>
      </div>

      {/* KaTeX Equations and Statistical Guarantees */}
      <div className="bg-[#060A12] border border-[#1B2536] rounded-xl p-3.5 flex flex-col gap-2.5 font-mono text-xs">
        <div className="text-[#E6EBF5] flex flex-wrap items-center gap-x-2 gap-y-1 text-xs">
          <span>Early exit gate condition:</span>
          <span className="bg-[#0E1420] px-2 py-0.5 rounded border border-[#1B2536] text-[#34D399]">
            <InlineMath math={`p_{1,0} \\ge \\tau_b`} />
          </span>
          <span className="text-[#7C8AA3]">
            (with configured threshold <InlineMath math={`\\tau_b = ${tau_b.toFixed(2)}`} />)
          </span>
        </div>

        <div className="text-[#E6EBF5] flex flex-wrap items-center gap-x-2 gap-y-1 text-xs">
          <span>Clopper-Pearson exact finite-sample bound:</span>
          <span className="bg-[#0E1420] px-2 py-0.5 rounded border border-[#1B2536] text-[#38BDF8]">
            <InlineMath math={`\\mathbb{P}(\\Delta R \\le \\epsilon) \\ge 1 - \\alpha = 0.95`} />
          </span>
        </div>

        <div className="text-[11px] text-[#7C8AA3] flex flex-wrap items-center justify-between pt-1 border-t border-[#1B2536]">
          <span>
            Target recall loss bound:{" "}
            <strong className="text-[#38BDF8]">
              <InlineMath math={`\\epsilon \\le ${epsilon.toFixed(3)}`} />
            </strong>
          </span>
          <span className="flex items-center gap-1.5">
            <span>Live measured recall loss:</span>
            <strong
              className={`${
                recallLoss <= epsilon ? "text-[#34D399]" : "text-[#F43F5E]"
              }`}
            >
              {recallLossPct}%
            </strong>
            <SourceTooltip source={stats?.recall_loss.source} />
          </span>
        </div>

        {/* Caveat */}
        <div className="mt-1 flex items-center gap-1.5 text-[10px] text-[#FBBF24]/90 bg-[#FBBF24]/5 border border-[#FBBF24]/20 px-2.5 py-1.5 rounded-lg">
          <AlertCircle className="w-3.5 h-3.5 shrink-0 text-[#FBBF24]" />
          <span>Caveat: Assumes calibration and test data are exchangeable.</span>
        </div>
      </div>
    </div>
  );
};
