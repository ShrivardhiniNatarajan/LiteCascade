import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { api, type LiveEvent, type LivePayload, type LiveSummary, type ModelCard } from "./api";

type LiveCtx = {
  cards: ModelCard[];
  cardsById: Record<string, ModelCard>;
  summary: LiveSummary | null;
  events: LiveEvent[];
  last: LiveEvent | null;
  running: boolean;
  speed: number;
  families: string[];
  active: string[];
  primary: string | null;
  connected: boolean;
  error: string | null;
  control: (action: "play" | "pause" | "reset" | "step", speed?: number) => Promise<void>;
  setSpeed: (s: number) => Promise<void>;
  inject: (family: string, windows?: number) => Promise<void>;
  setActive: (ids: string[]) => Promise<void>;
};

const Ctx = createContext<LiveCtx | null>(null);
const HISTORY = 360;

export function LiveProvider({ children }: { children: ReactNode }) {
  const [cards, setCards] = useState<ModelCard[]>([]);
  const [summary, setSummary] = useState<LiveSummary | null>(null);
  const [events, setEvents] = useState<LiveEvent[]>([]);
  const [running, setRunning] = useState(false);
  const [speed, setSpeedState] = useState(10);
  const [families, setFamilies] = useState<string[]>([]);
  const [active, setActiveState] = useState<string[]>([]);
  const [primary, setPrimary] = useState<string | null>(null);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const buf = useRef<LiveEvent[]>([]);
  const pending = useRef<LivePayload | null>(null);

  const apply = useCallback((p: LivePayload) => {
    if (p.event) {
      buf.current = [...buf.current.slice(-(HISTORY - 1)), p.event];
    }
    pending.current = p;
  }, []);

  // Batch stream updates into animation frames so fast replay speeds stay smooth.
  useEffect(() => {
    let raf = 0;
    const tick = () => {
      const p = pending.current;
      if (p) {
        pending.current = null;
        setSummary(p.summary);
        setRunning(p.running);
        setSpeedState(p.speed);
        setEvents(buf.current);
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  useEffect(() => {
    api.get<ModelCard[]>("/api/models").then(setCards).catch((e) => setError(String(e.message ?? e)));
    api
      .get<LivePayload & { families: string[]; active: string[]; primary: string }>("/api/live/state")
      .then((s) => {
        apply(s);
        setFamilies(s.families);
        setActiveState(s.active);
        setPrimary(s.primary);
      })
      .catch((e) => setError(String(e.message ?? e)));
    const es = new EventSource("/api/live/stream");
    es.onopen = () => setConnected(true);
    es.onerror = () => setConnected(false);
    es.onmessage = (m) => apply(JSON.parse(m.data));
    return () => es.close();
  }, [apply]);

  const control = useCallback(async (action: "play" | "pause" | "reset" | "step", s?: number) => {
    const body: Record<string, unknown> = { action };
    if (s != null) body.speed = s;
    const p = await api.post<LivePayload>("/api/live/control", body);
    if (action === "reset") buf.current = [];
    apply(p);
  }, [apply]);

  const setSpeed = useCallback(async (s: number) => {
    const p = await api.post<LivePayload>("/api/live/control", { speed: s });
    apply(p);
  }, [apply]);

  const inject = useCallback(async (family: string, windows = 30) => {
    await api.post("/api/live/inject", { family, windows });
  }, []);

  const setActive = useCallback(async (ids: string[]) => {
    const r = await api.post<{ active: string[] }>("/api/live/models", { active: ids });
    setActiveState(r.active);
  }, []);

  const cardsById = useMemo(() => Object.fromEntries(cards.map((c) => [c.id, c])), [cards]);
  const value: LiveCtx = {
    cards, cardsById, summary, events, last: events.length ? events[events.length - 1] : null, running, speed,
    families, active, primary, connected, error, control, setSpeed, inject, setActive,
  };
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useLive() {
  const v = useContext(Ctx);
  if (!v) throw new Error("useLive outside LiveProvider");
  return v;
}
