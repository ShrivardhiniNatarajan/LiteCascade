import React, { useEffect, useRef } from "react";
import { Terminal, AlertTriangle, Pause } from "lucide-react";
import { useStore } from "../store/useStore";
import { SourceTooltip } from "./SourceTooltip";

export const InferenceLogCard: React.FC = () => {
  const { meta, events, isLogPaused, setIsLogPaused } = useStore();
  const logContainerRef = useRef<HTMLDivElement>(null);

  const epsilon = meta?.epsilon.value ?? 0.05;
  const isGateLoaded = meta?.gate_loaded.value ?? false;

  useEffect(() => {
    if (!isLogPaused && logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [events, isLogPaused]);

  return (
    <div className="w-full bg-[#0E1420] border border-[#1B2536] rounded-card p-5 relative overflow-hidden transition-all shadow-sm flex flex-col h-[320px]">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-[#1B2536]">
        <div className="flex items-center gap-2">
          <Terminal className="w-4 h-4 text-[#38BDF8]" />
          <h2 className="text-xs font-mono font-bold tracking-wider uppercase text-[#E6EBF5]">
            5. Real-Time Inference Log
          </h2>
          <SourceTooltip source={meta?.epsilon.source} />
        </div>

        {/* Header Badges */}
        <div className="flex items-center gap-2">
          {isGateLoaded ? (
            <span
              className="text-[10px] font-mono bg-[#060A12] border border-[#1B2536] px-2 py-0.5 rounded text-[#7C8AA3]"
              title="Clopper-Pearson Exact 95% Confidence Upper Bound"
            >
              Clopper-Pearson bound:{" "}
              <strong className="text-[#38BDF8]">ε ≤ {epsilon.toFixed(3)}</strong>
            </span>
          ) : (
            <span className="flex items-center gap-1 text-[10px] font-mono bg-[#FBBF24]/10 border border-[#FBBF24]/40 text-[#FBBF24] px-2 py-0.5 rounded font-bold animate-pulse">
              <AlertTriangle className="w-3 h-3 text-[#FBBF24]" />
              <span>Gate Not Loaded</span>
            </span>
          )}

          {isLogPaused && (
            <span className="flex items-center gap-1 text-[10px] font-mono text-[#FBBF24] bg-[#060A12] px-2 py-0.5 rounded border border-[#1B2536]">
              <Pause className="w-2.5 h-2.5" /> Paused
            </span>
          )}
        </div>
      </div>

      {/* Log Feed Container */}
      <div
        ref={logContainerRef}
        onMouseEnter={() => setIsLogPaused(true)}
        onMouseLeave={() => setIsLogPaused(false)}
        className="flex-1 overflow-y-auto mt-2 font-mono text-xs dark-scrollbar space-y-1.5 pr-1 select-text"
      >
        {events.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-[#7C8AA3] text-xs font-mono space-y-2 opacity-60">
            <Terminal className="w-6 h-6 text-[#7C8AA3]" />
            <p>Stream idle. Click &quot;Resume Stream&quot; or press Space to stream inference windows.</p>
          </div>
        ) : (
          events.map((ev) => {
            const isAlert = ev.y_pred === 1 || ev.is_injected;
            const isEscalatedVerified = ev.exit === "analyst" && !isAlert;

            const timeStr = ev.timestamp ? ev.timestamp.split("T")[1]?.slice(0, 8) : "--:--:--";

            return (
              <div
                key={ev.idx}
                className={`py-1 px-2 rounded flex items-center justify-between transition-colors text-[11px] leading-relaxed border ${
                  isAlert
                    ? "bg-[#F43F5E]/10 text-[#F43F5E] border-[#F43F5E]/30"
                    : isEscalatedVerified
                    ? "bg-[#38BDF8]/5 text-[#38BDF8] border-[#38BDF8]/20"
                    : "bg-[#34D399]/5 text-[#34D399] border-[#34D399]/20"
                }`}
              >
                <div className="flex items-center gap-2 overflow-hidden">
                  <span className="text-[#7C8AA3] text-[10px] select-none font-light">
                    [{timeStr}]
                  </span>
                  <span className="font-semibold text-white/90">
                    #{ev.idx.toString().padStart(4, "0")}
                  </span>
                  <span className="text-white/60 truncate max-w-[120px]" title={ev.flow_id}>
                    {ev.flow_id}
                  </span>
                  <span className="opacity-40 select-none">|</span>
                  <span className="font-bold">
                    {ev.exit === "sentinel" ? "E1:Sentinel" : "E2:Analyst"}
                  </span>
                  <span className="opacity-40 select-none">|</span>
                  <span>
                    Conf: {(ev.confidence * 100).toFixed(1)}%
                  </span>
                </div>

                <div className="flex items-center gap-2 shrink-0">
                  <span
                    className={`px-1.5 py-0.2 rounded text-[10px] font-bold uppercase tracking-wider ${
                      isAlert
                        ? "bg-[#F43F5E] text-white"
                        : isEscalatedVerified
                        ? "bg-[#38BDF8]/20 text-[#38BDF8]"
                        : "bg-[#34D399]/20 text-[#34D399]"
                    }`}
                  >
                    {isAlert ? `ALERT: ${ev.family_pred}` : ev.sentinel_verdict === "Uncertain" ? "Escalated Benign" : "Benign"}
                  </span>
                  <span className="text-[#7C8AA3] text-[10px] min-w-[45px] text-right">
                    {ev.latency_ms.toFixed(2)}ms
                  </span>
                </div>
              </div>
            );
          })
        )}
      </div>

      {/* Footer hint */}
      <div className="pt-2 border-t border-[#1B2536] flex items-center justify-between text-[10px] text-[#7C8AA3] font-mono">
        <span>Hover log to pause auto-scroll</span>
        <span>Showing last {events.length} / 100 events</span>
      </div>
    </div>
  );
};
