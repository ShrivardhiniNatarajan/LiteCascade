import { motion } from "framer-motion";
import { Activity, BookOpen, Boxes, FileSearch, Gauge, Moon, Sun, Swords, Trophy } from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";
import { useLive } from "./lib/live";
import { AnalyzePage } from "./pages/Analyze";
import { ArchitecturePage } from "./pages/Architecture";
import { ArenaPage } from "./pages/Arena";
import { BenchmarksPage } from "./pages/Benchmarks";
import { DocsPage } from "./pages/Docs";
import { GateLabPage } from "./pages/GateLab";
import { LivePage } from "./pages/Live";

const PAGES: { id: string; label: string; icon: ReactNode; el: () => ReactNode }[] = [
  { id: "live", label: "Live cascade", icon: <Activity size={17} />, el: () => <LivePage /> },
  { id: "arena", label: "Model arena", icon: <Swords size={17} />, el: () => <ArenaPage /> },
  { id: "bench", label: "Benchmarks", icon: <Trophy size={17} />, el: () => <BenchmarksPage /> },
  { id: "gate", label: "Gate lab", icon: <Gauge size={17} />, el: () => <GateLabPage /> },
  { id: "arch", label: "Architecture", icon: <Boxes size={17} />, el: () => <ArchitecturePage /> },
  { id: "analyze", label: "Analyze CSV", icon: <FileSearch size={17} />, el: () => <AnalyzePage /> },
  { id: "docs", label: "Documentation", icon: <BookOpen size={17} />, el: () => <DocsPage /> },
];

function Logo() {
  return (
    <motion.svg width="34" height="34" viewBox="0 0 32 32" animate={{ rotate: [0, 6, -6, 0] }} transition={{ repeat: Infinity, duration: 8 }}>
      <path d="M16 2 29 9v14l-13 7-13-7V9z" fill="var(--sentinel)" />
      <motion.path d="M16 9 23 13v7l-7 4-7-4v-7z" fill="var(--analyst)" animate={{ opacity: [1, 0.55, 1] }} transition={{ repeat: Infinity, duration: 2.4 }} />
    </motion.svg>
  );
}

export default function App() {
  const initial = (location.hash.replace("#", "") || "live").split("?")[0];
  const [page, setPage] = useState(PAGES.some((p) => p.id === initial) ? initial : "live");
  const [theme, setTheme] = useState<"dark" | "light">(() => {
    try {
      return (localStorage.getItem("lc-theme") as "dark" | "light") || "dark";
    } catch {
      return "dark";
    }
  });
  const { connected, cards, error, running } = useLive();
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem("lc-theme", theme);
    } catch {
      /* storage unavailable */
    }
  }, [theme]);
  useEffect(() => {
    if (location.hash.replace("#", "") !== page) location.hash = page;
  }, [page]);
  useEffect(() => {
    const onHash = () => {
      const h = location.hash.replace("#", "");
      if (PAGES.some((p) => p.id === h)) setPage(h);
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  const synthetic = cards.some((c) => c.is_synthetic);
  const current = PAGES.find((p) => p.id === page)!;

  return (
    <div className="shell">
      <nav className="rail">
        <div className="brand">
          <Logo />
          <div>
            <h1>LiteCascade</h1>
            <p>IoT intrusion control room</p>
          </div>
        </div>
        {PAGES.map((p) => (
          <button key={p.id} className={`navbtn ${page === p.id ? "active" : ""}`} onClick={() => setPage(p.id)}>
            {page === p.id && <motion.span layoutId="nav-bg" className="pill-bg" transition={{ type: "spring", stiffness: 400, damping: 32 }} />}
            {p.icon}
            <span>{p.label}</span>
          </button>
        ))}
        <div className="rail-foot">
          <span style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <motion.span style={{ width: 8, height: 8, borderRadius: 9, background: connected ? "var(--ok)" : "var(--malicious)" }}
              animate={running ? { opacity: [1, 0.3, 1] } : { opacity: 1 }} transition={{ repeat: Infinity, duration: 1 }} />
            {connected ? (running ? "streaming" : "connected") : "backend offline"}
          </span>
          <span>{cards.length} models loaded</span>
          {synthetic && <span className="badge syn">synthetic fixture data</span>}
          <button className="btn" onClick={() => setTheme(theme === "dark" ? "light" : "dark")} style={{ justifyContent: "center" }}>
            {theme === "dark" ? <Sun size={15} /> : <Moon size={15} />} {theme === "dark" ? "Light" : "Dark"} theme
          </button>
        </div>
      </nav>
      <main className="main">
        {error && <div className="status" style={{ borderColor: "var(--malicious)" }}>Backend error: {error}</div>}
        {/* Enter-only page transition: an exit animation (AnimatePresence mode="wait") can stall while the live
            stream re-renders the outgoing page every frame, which would block navigation. */}
        <motion.div key={page} initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.25 }}>
          {current.el()}
        </motion.div>
      </main>
    </div>
  );
}
