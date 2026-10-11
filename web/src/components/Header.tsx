import React from "react";
import { Play, Pause, StepForward, RotateCcw, ShieldAlert, Cpu } from "lucide-react";
import { useStore } from "../store/useStore";
import { SourceTooltip } from "./SourceTooltip";
import logoImg from "../assets/logo.png";

export const Header: React.FC = () => {
  const {
    meta,
    lastEvent,
    isRunning,
    speed,
    toggleStream,
    stepStream,
    resetStream,
    setSpeed,
    openInject,
  } = useStore();

  const isAlert = lastEvent ? (lastEvent.y_pred === 1 || lastEvent.is_injected) : false;
  const isEscalated = lastEvent ? lastEvent.exit === "analyst" : false;

  // Real status state: red = alert, amber = elevated analyst exit, green = clean normal
  const statusColor = isAlert ? "#F43F5E" : isEscalated ? "#FBBF24" : "#34D399";
  const statusText = isAlert
    ? `INTRUSION ALERT (${(lastEvent?.family_pred || "MALICIOUS").toUpperCase()})`
    : isEscalated
    ? "EVALUATING AT EXIT 2"
    : "NORMAL STREAM";

  const isSynthetic = meta?.is_synthetic.value ?? true;
  const dataLabel = isSynthetic ? "Synthetic fixture" : "N-BaIoT test split";
  const modelFormat = meta?.model_format.value ?? "PyTorch FP32";

  return (
    <header className="w-full bg-[#0E1420] border-b border-[#1B2536] px-4 lg:px-8 py-3.5 shadow-lg">
      <div className="max-w-[1280px] mx-auto flex flex-col xl:flex-row xl:items-center xl:justify-between gap-4">
        {/* Left Side: AI Logo + LITECASCADE Branding & Telemetry Pills */}
        <div className="flex flex-wrap items-center gap-4 lg:gap-6">
          {/* Brand Identity with AI Generated Shield/Neural Logo */}
          <div className="flex items-center gap-3">
            <div className="relative group shrink-0">
              <div className="absolute -inset-0.5 bg-gradient-to-r from-[#34D399] to-[#38BDF8] rounded-2xl blur-[2px] opacity-75 group-hover:opacity-100 transition duration-300"></div>
              <img
                src={logoImg}
                alt="LiteCascade AI Logo"
                className="relative w-11 h-11 rounded-2xl object-cover border border-[#1B2536] bg-[#070B12] shadow-inner"
              />
            </div>

            <div className="flex flex-col">
              <div className="flex items-center gap-2">
                <h1 className="text-xl font-extrabold tracking-tight text-white flex items-center">
                  <span>Lite</span>
                  <span className="text-[#34D399]">Cascade</span>
                </h1>
                <span className="text-[10px] font-mono uppercase tracking-widest px-1.5 py-0.5 rounded bg-[#1B2536] text-[#7C8AA3] border border-[#1B2536] font-bold">
                  SIM
                </span>
              </div>
              <p className="text-[11px] text-[#7C8AA3] font-medium leading-tight hidden sm:block">
                Risk-Controlled Early-Exit Simulation for Resource-Constrained IoT Hardware
              </p>
            </div>
          </div>

          {/* Status & Provenance Badges */}
          <div className="flex items-center gap-2.5 flex-wrap">
            {/* Pulsing Status Dot */}
            <div className="flex items-center gap-2 bg-[#060A12] border border-[#1B2536] px-3 py-1 rounded-full shadow-sm">
              <span className="relative flex h-2.5 w-2.5">
                <span
                  className="animate-ping absolute inline-flex h-full w-full rounded-full opacity-75"
                  style={{ backgroundColor: statusColor }}
                ></span>
                <span
                  className="relative inline-flex rounded-full h-2.5 w-2.5"
                  style={{ backgroundColor: statusColor }}
                ></span>
              </span>
              <span
                className="text-xs font-mono font-bold tracking-wide uppercase"
                style={{ color: statusColor }}
              >
                {statusText}
              </span>
            </div>

            {/* Model Format Pill */}
            <div className="flex items-center gap-1.5 bg-[#060A12] border border-[#1B2536] px-2.5 py-1 rounded-full text-xs font-mono text-[#38BDF8]">
              <Cpu className="w-3.5 h-3.5 text-[#38BDF8]" />
              <span>{modelFormat}</span>
              <SourceTooltip source={meta?.model_format.source} />
            </div>

            {/* Provenance Pill */}
            <div
              className={`flex items-center gap-1.5 border px-2.5 py-1 rounded-full text-xs font-mono font-medium ${
                isSynthetic
                  ? "bg-[#FBBF24]/10 border-[#FBBF24]/30 text-[#FBBF24]"
                  : "bg-[#34D399]/10 border-[#34D399]/30 text-[#34D399]"
              }`}
            >
              <span>{dataLabel}</span>
              <SourceTooltip source={meta?.data_source.source} />
            </div>
          </div>
        </div>

        {/* Right Side: Interactive Controls & Speed */}
        <div className="flex items-center gap-2.5 flex-wrap justify-end">
          {/* Speed Selector */}
          <div className="flex items-center bg-[#060A12] border border-[#1B2536] rounded-xl p-0.5 text-xs font-mono">
            {[1, 10, 100].map((s) => (
              <button
                key={s}
                onClick={() => setSpeed(s)}
                className={`px-2.5 py-1 rounded-lg transition-all ${
                  speed === s
                    ? "bg-[#1B2536] text-[#34D399] font-bold shadow-sm"
                    : "text-[#7C8AA3] hover:text-[#E6EBF5]"
                }`}
                title={`Playback speed ${s}x`}
              >
                {s}x
              </button>
            ))}
          </div>

          {/* Secondary Action: Step */}
          <button
            onClick={stepStream}
            className="p-2 rounded-xl bg-[#060A12] border border-[#1B2536] text-[#E6EBF5] hover:border-[#38BDF8] hover:text-[#38BDF8] transition-all flex items-center justify-center"
            title="Step 1 window (S)"
          >
            <StepForward className="w-4 h-4" />
          </button>

          {/* Secondary Action: Reset */}
          <button
            onClick={resetStream}
            className="p-2 rounded-xl bg-[#060A12] border border-[#1B2536] text-[#E6EBF5] hover:border-[#F43F5E] hover:text-[#F43F5E] transition-all flex items-center justify-center"
            title="Reset telemetry (R)"
          >
            <RotateCcw className="w-4 h-4" />
          </button>

          {/* Primary Action: Stream Play/Pause */}
          <button
            onClick={toggleStream}
            className={`px-4 py-2 rounded-xl text-xs font-bold font-mono tracking-wide flex items-center gap-2 transition-all shadow-glowGreen ${
              isRunning
                ? "bg-[#34D399] text-[#070B12] hover:bg-[#34D399]/90"
                : "bg-[#34D399] text-[#070B12] hover:bg-[#34D399]/90"
            }`}
            title="Toggle stream (Space)"
          >
            {isRunning ? (
              <>
                <Pause className="w-3.5 h-3.5 fill-current" />
                <span>Pause Stream</span>
              </>
            ) : (
              <>
                <Play className="w-3.5 h-3.5 fill-current" />
                <span>Resume Stream</span>
              </>
            )}
          </button>

          {/* Botnet Injection Trigger */}
          <button
            onClick={() => openInject(true)}
            className="px-4 py-2 rounded-xl text-xs font-bold font-mono tracking-wide flex items-center gap-2 bg-[#F43F5E] text-white hover:bg-[#F43F5E]/90 transition-all shadow-glowRed"
            title="Inject Botnet Flow (A)"
          >
            <ShieldAlert className="w-3.5 h-3.5" />
            <span>Inject Botnet Flow</span>
          </button>
        </div>
      </div>
    </header>
  );
};
