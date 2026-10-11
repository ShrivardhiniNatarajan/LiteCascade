import React from "react";
import { X, ShieldAlert, AlertTriangle, CheckCircle2 } from "lucide-react";
import { useStore } from "../store/useStore";

export const InjectModal: React.FC = () => {
  const {
    meta,
    isInjectOpen,
    injectFamily,
    injectDuration,
    injectLoading,
    injectError,
    injectSuccess,
    openInject,
    setInjectFamily,
    setInjectDuration,
    submitInject,
  } = useStore();

  if (!isInjectOpen) return null;

  const availableFamilies =
    meta?.available_families.value && meta.available_families.value.length > 0
      ? meta.available_families.value.filter((f: string) => f.toLowerCase() !== "benign")
      : ["Mirai", "Gafgyt", "Tsunami", "DDoS_SYN"];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-fade-in">
      <div className="w-full max-w-md bg-[#0E1420] border border-[#1B2536] rounded-card shadow-2xl p-6 relative flex flex-col gap-4">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-[#F43F5E]">
            <ShieldAlert className="w-5 h-5" />
            <h3 className="font-bold text-sm text-[#E6EBF5] font-mono tracking-wide">
              Inject Simulated Botnet Segment
            </h3>
          </div>
          <button
            onClick={() => openInject(false)}
            className="text-[#7C8AA3] hover:text-[#E6EBF5] p-1 rounded-lg hover:bg-[#1B2536] transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <p className="text-xs text-[#7C8AA3]">
          Injects empirical malware flow characteristics into consecutive streaming windows to evaluate risk gate
          escalation and detection recall.
        </p>

        {/* Feedback Messages */}
        {injectError && (
          <div className="bg-[#F43F5E]/10 border border-[#F43F5E]/40 text-[#F43F5E] p-3 rounded-xl text-xs flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 shrink-0" />
            <span>{injectError}</span>
          </div>
        )}

        {injectSuccess && (
          <div className="bg-[#34D399]/10 border border-[#34D399]/40 text-[#34D399] p-3 rounded-xl text-xs flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 shrink-0" />
            <span>{injectSuccess}</span>
          </div>
        )}

        {/* Form Fields */}
        <div className="flex flex-col gap-3 font-mono text-xs">
          <div>
            <label className="block text-[#7C8AA3] mb-1.5">Malware Family</label>
            <div className="grid grid-cols-2 gap-2">
              {availableFamilies.map((fam: string) => (
                <button
                  key={fam}
                  type="button"
                  onClick={() => setInjectFamily(fam)}
                  className={`p-2.5 rounded-xl border text-left transition-all ${
                    injectFamily.toLowerCase() === fam.toLowerCase()
                      ? "bg-[#F43F5E]/10 border-[#F43F5E] text-[#F43F5E] font-bold"
                      : "bg-[#060A12] border-[#1B2536] text-[#7C8AA3] hover:border-[#7C8AA3] hover:text-[#E6EBF5]"
                  }`}
                >
                  {fam}
                </button>
              ))}
            </div>
          </div>

          <div>
            <div className="flex items-center justify-between mb-1.5">
              <label className="text-[#7C8AA3]">Duration (Windows)</label>
              <span className="text-[#38BDF8] font-bold">{injectDuration} windows</span>
            </div>
            <input
              type="range"
              min="5"
              max="50"
              step="5"
              value={injectDuration}
              onChange={(e) => setInjectDuration(Number(e.target.value))}
              className="w-full accent-[#F43F5E] bg-[#1B2536] rounded-lg h-1.5 cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-[#7C8AA3] mt-1">
              <span>5 windows (~0.5s)</span>
              <span>50 windows (~5s)</span>
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center justify-end gap-2.5 mt-2 font-mono text-xs">
          <button
            type="button"
            onClick={() => openInject(false)}
            className="px-4 py-2 rounded-xl bg-[#060A12] border border-[#1B2536] text-[#7C8AA3] hover:text-[#E6EBF5] transition-colors"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={submitInject}
            disabled={injectLoading}
            className="px-4 py-2 rounded-xl bg-[#F43F5E] text-white font-bold hover:bg-[#F43F5E]/90 transition-all flex items-center gap-2 shadow-glowRed disabled:opacity-50"
          >
            {injectLoading ? "Scheduling..." : "Inject Attack Segment"}
          </button>
        </div>
      </div>
    </div>
  );
};
