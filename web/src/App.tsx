import React, { useEffect } from "react";
import { useStore } from "./store/useStore";
import { Header } from "./components/Header";
import { InputIngestCard } from "./components/InputIngestCard";
import { PipStemCard } from "./components/PipStemCard";
import { SentinelCard } from "./components/SentinelCard";
import { AnalystCard } from "./components/AnalystCard";
import { PipelineConnector } from "./components/PipelineConnector";
import { InferenceLogCard } from "./components/InferenceLogCard";
import { EdgeTelemetryCard } from "./components/EdgeTelemetryCard";
import { SafetyNetCard } from "./components/SafetyNetCard";
import { InjectModal } from "./components/InjectModal";
import { Footer } from "./components/Footer";

export const App: React.FC = () => {
  const {
    init,
    toggleStream,
    stepStream,
    resetStream,
    openInject,
    isInjectOpen,
    lastEvent,
  } = useStore();

  useEffect(() => {
    init();
  }, [init]);

  // Global keyboard shortcuts: Space, S, A, R
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't trigger shortcuts if user is typing in an input / select
      const tag = (e.target as HTMLElement)?.tagName?.toLowerCase();
      if (tag === "input" || tag === "select" || tag === "textarea") {
        return;
      }

      if (e.code === "Space") {
        e.preventDefault();
        toggleStream();
      } else if (e.key === "s" || e.key === "S") {
        e.preventDefault();
        stepStream();
      } else if (e.key === "a" || e.key === "A") {
        e.preventDefault();
        openInject(!isInjectOpen);
      } else if (e.key === "r" || e.key === "R") {
        e.preventDefault();
        resetStream();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [toggleStream, stepStream, resetStream, openInject, isInjectOpen]);

  const isEscalated = lastEvent?.exit === "analyst";

  return (
    <div className="min-h-screen bg-[#070B12] text-[#E6EBF5] flex flex-col selection:bg-[#34D399]/20 selection:text-[#34D399]">
      <Header />

      <main className="flex-1 max-w-[1280px] w-full mx-auto px-4 lg:px-8 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          {/* Left Column: Cascade Pipeline Architecture (5 cols on lg) */}
          <div className="lg:col-span-5 flex flex-col">
            <InputIngestCard />

            <PipelineConnector label="d=16 features" active={true} />

            <PipStemCard />

            <PipelineConnector label="c=24 channels" active={true} />

            <SentinelCard />

            <PipelineConnector
              label={isEscalated ? "Escalated to Exit 2" : "Exit 1 granted (bypassed)"}
              active={isEscalated}
            />

            <AnalystCard />
          </div>

          {/* Right Column: Real-time Telemetry & Intelligence (7 cols on lg) */}
          <div className="lg:col-span-7 flex flex-col gap-6">
            <InferenceLogCard />

            <EdgeTelemetryCard />

            <SafetyNetCard />
          </div>
        </div>

        <Footer />
      </main>

      <InjectModal />
    </div>
  );
};

export default App;
