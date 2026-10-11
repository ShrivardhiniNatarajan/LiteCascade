import React from "react";
import { GitBranch } from "lucide-react";
import { useStore } from "../store/useStore";
import { SourceTooltip } from "./SourceTooltip";

export const Footer: React.FC = () => {
  const { meta } = useStore();
  const gitCommit = meta?.git_commit.value ? meta.git_commit.value.slice(0, 8) : "HEAD";

  return (
    <footer className="w-full py-6 mt-6 border-t border-[#1B2536] flex flex-col sm:flex-row items-center justify-between text-xs text-[#7C8AA3] font-mono gap-3">
      <div>
        <span>Research demonstration on replayed public datasets, not a production IDS</span>
      </div>
      <div className="flex items-center gap-2">
        <GitBranch className="w-3.5 h-3.5 text-[#34D399]" />
        <span>Commit:</span>
        <code className="text-[#E6EBF5] bg-[#060A12] px-2 py-0.5 rounded border border-[#1B2536]">
          {gitCommit}
        </code>
        <SourceTooltip source={meta?.git_commit.source} />
      </div>
    </footer>
  );
};
