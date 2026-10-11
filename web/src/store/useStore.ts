import { create } from "zustand";
import {
  type MetaResponse,
  type StatsResponse,
  type StreamEvent,
  fetchMeta,
  fetchStats,
  fetchDevice,
  controlStream,
  injectAttack,
} from "../api/client";

export interface DashboardState {
  meta: MetaResponse | null;
  stats: StatsResponse | null;
  device: Record<string, any> | null;
  events: StreamEvent[];
  lastEvent: StreamEvent | null;
  isRunning: boolean;
  speed: number;
  isInjectOpen: boolean;
  injectFamily: string;
  injectDuration: number;
  injectLoading: boolean;
  injectError: string | null;
  injectSuccess: string | null;
  isLogPaused: boolean;
  eventSource: EventSource | null;

  // Actions
  init: () => Promise<void>;
  startStream: () => Promise<void>;
  pauseStream: () => Promise<void>;
  toggleStream: () => Promise<void>;
  stepStream: () => Promise<void>;
  resetStream: () => Promise<void>;
  setSpeed: (speed: number) => Promise<void>;
  openInject: (open: boolean) => void;
  setInjectFamily: (family: string) => void;
  setInjectDuration: (duration: number) => void;
  submitInject: () => Promise<void>;
  setIsLogPaused: (paused: boolean) => void;
  addEvent: (event: StreamEvent) => void;
}

export const useStore = create<DashboardState>((set, get) => ({
  meta: null,
  stats: null,
  device: null,
  events: [],
  lastEvent: null,
  isRunning: false,
  speed: 10,
  isInjectOpen: false,
  injectFamily: "Mirai",
  injectDuration: 15,
  injectLoading: false,
  injectError: null,
  injectSuccess: null,
  isLogPaused: false,
  eventSource: null,

  init: async () => {
    try {
      const [meta, stats, device] = await Promise.all([
        fetchMeta(),
        fetchStats(),
        fetchDevice(),
      ]);
      set({ meta, stats, device });
    } catch (e) {
      console.error("Initialization error:", e);
    }

    // Connect to Server-Sent Events stream
    if (typeof window !== "undefined" && !get().eventSource) {
      const es = new EventSource("/api/stream/events");
      es.onmessage = (msg) => {
        try {
          const ev: StreamEvent = JSON.parse(msg.data);
          get().addEvent(ev);
        } catch (e) {
          console.error("Error parsing SSE event:", e);
        }
      };
      es.onerror = (err) => {
        console.warn("SSE connection interrupted, retrying...", err);
      };
      set({ eventSource: es });
    }
  },

  addEvent: (ev: StreamEvent) => {
    const current = get().events;
    const updated = [...current, ev].slice(-100);
    set({ events: updated, lastEvent: ev });

    // Periodically sync stats
    if (ev.idx % 5 === 0) {
      fetchStats()
        .then((s) => set({ stats: s }))
        .catch(() => {});
    }
  },

  startStream: async () => {
    try {
      const res = await controlStream("start", get().speed);
      set({ isRunning: res.is_running });
    } catch (e) {
      console.error("Start stream failed:", e);
    }
  },

  pauseStream: async () => {
    try {
      const res = await controlStream("pause");
      set({ isRunning: res.is_running });
    } catch (e) {
      console.error("Pause stream failed:", e);
    }
  },

  toggleStream: async () => {
    if (get().isRunning) {
      await get().pauseStream();
    } else {
      await get().startStream();
    }
  },

  stepStream: async () => {
    try {
      await controlStream("step");
      const stats = await fetchStats();
      set({ stats });
    } catch (e) {
      console.error("Step stream failed:", e);
    }
  },

  resetStream: async () => {
    try {
      await controlStream("reset");
      const stats = await fetchStats();
      set({ events: [], lastEvent: null, isRunning: false, stats });
    } catch (e) {
      console.error("Reset stream failed:", e);
    }
  },

  setSpeed: async (speed: number) => {
    set({ speed });
    if (get().isRunning) {
      try {
        await controlStream("start", speed);
      } catch (e) {
        console.error("Speed update failed:", e);
      }
    }
  },

  openInject: (open: boolean) => {
    set({ isInjectOpen: open, injectError: null, injectSuccess: null });
  },

  setInjectFamily: (injectFamily: string) => set({ injectFamily }),
  setInjectDuration: (injectDuration: number) => set({ injectDuration }),

  submitInject: async () => {
    const { injectFamily, injectDuration } = get();
    set({ injectLoading: true, injectError: null, injectSuccess: null });
    try {
      const res = await injectAttack({
        family: injectFamily,
        duration_windows: injectDuration,
      });
      set({
        injectLoading: false,
        injectSuccess: `Scheduled ${res.family} (${res.duration_windows} windows)!`,
      });
      setTimeout(() => set({ isInjectOpen: false, injectSuccess: null }), 1400);
    } catch (e: any) {
      set({ injectLoading: false, injectError: e.message || "Injection failed" });
    }
  },

  setIsLogPaused: (isLogPaused: boolean) => set({ isLogPaused }),
}));
