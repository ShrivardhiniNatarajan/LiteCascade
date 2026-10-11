import React from "react";
import { ChevronDown } from "lucide-react";

interface PipelineConnectorProps {
  label?: string;
  active?: boolean;
}

export const PipelineConnector: React.FC<PipelineConnectorProps> = ({
  label,
  active = false,
}) => {
  return (
    <div className="flex flex-col items-center justify-center my-0.5 py-0.5 relative select-none">
      <div className="h-4 w-px bg-gradient-to-b from-[#1B2536] via-[#38BDF8]/40 to-[#1B2536]" />
      <div
        className={`flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[10px] font-mono tracking-wider transition-all ${
          active
            ? "bg-[#060A12] border border-[#38BDF8]/50 text-[#38BDF8]"
            : "bg-[#060A12]/80 border border-[#1B2536] text-[#7C8AA3]"
        }`}
      >
        <ChevronDown
          className={`w-3 h-3 ${active ? "text-[#38BDF8] animate-bounce" : "text-[#7C8AA3]"}`}
        />
        {label && <span>{label}</span>}
      </div>
      <div className="h-4 w-px bg-gradient-to-b from-[#1B2536] via-[#38BDF8]/40 to-[#1B2536]" />
    </div>
  );
};
