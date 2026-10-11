import React from "react";
import { Info } from "lucide-react";

interface SourceTooltipProps {
  source?: string;
  className?: string;
}

export const SourceTooltip: React.FC<SourceTooltipProps> = ({ source, className = "" }) => {
  if (!source) return null;

  return (
    <span className={`group relative inline-flex items-center text-xs ml-1 cursor-help ${className}`}>
      <Info className="w-3 h-3 text-[#7C8AA3] group-hover:text-[#38BDF8] transition-colors" />
      <span className="pointer-events-none absolute bottom-full left-1/2 -translate-x-1/2 mb-1.5 hidden group-hover:flex flex-col items-center z-50 whitespace-nowrap">
        <span className="bg-[#060A12] border border-[#1B2536] text-[#E6EBF5] text-[10px] font-mono px-2 py-0.5 rounded shadow-lg">
          source: {source}
        </span>
        <span className="w-1.5 h-1.5 bg-[#060A12] border-b border-r border-[#1B2536] rotate-45 -mt-1"></span>
      </span>
    </span>
  );
};
