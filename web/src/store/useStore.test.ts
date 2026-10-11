import { describe, it, expect, beforeEach } from "vitest";
import { useStore } from "./useStore";
import type { StreamEvent } from "../api/client";

describe("Dashboard Zustand Store", () => {
  beforeEach(() => {
    // Reset store state before each test
    useStore.setState({
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
    });
  });

  it("initializes with expected default values", () => {
    const state = useStore.getState();
    expect(state.isRunning).toBe(false);
    expect(state.speed).toBe(10);
    expect(state.events).toHaveLength(0);
    expect(state.lastEvent).toBeNull();
    expect(state.isInjectOpen).toBe(false);
    expect(state.injectFamily).toBe("Mirai");
    expect(state.injectDuration).toBe(15);
  });

  it("adds events and updates lastEvent properly", () => {
    const mockEvent: StreamEvent = {
      idx: 1,
      flow_id: "flow_test_001",
      y_true: 0,
      y_pred: 0,
      family_pred: "benign",
      confidence: 0.98,
      tau_b_used: 0.90,
      exit: "sentinel",
      sentinel_verdict: "Benign",
      analyst_verdict: null,
      latency_ms: 0.85,
      macs_used: 0.008,
      is_injected: false,
      timestamp: new Date().toISOString(),
    };

    useStore.getState().addEvent(mockEvent);

    const state = useStore.getState();
    expect(state.events).toHaveLength(1);
    expect(state.events[0].flow_id).toBe("flow_test_001");
    expect(state.lastEvent).toEqual(mockEvent);
  });

  it("caps events queue at 100 items", () => {
    for (let i = 0; i < 110; i++) {
      const ev: StreamEvent = {
        idx: i,
        flow_id: `flow_${i}`,
        y_true: 0,
        y_pred: 0,
        family_pred: "benign",
        confidence: 0.95,
        tau_b_used: 0.90,
        exit: "sentinel",
        sentinel_verdict: "Benign",
        analyst_verdict: null,
        latency_ms: 0.8,
        macs_used: 0.008,
        is_injected: false,
        timestamp: new Date().toISOString(),
      };
      useStore.getState().addEvent(ev);
    }

    const state = useStore.getState();
    expect(state.events).toHaveLength(100);
    expect(state.events[0].idx).toBe(10); // First 10 rolled off
    expect(state.events[99].idx).toBe(109);
    expect(state.lastEvent?.idx).toBe(109);
  });

  it("handles injection modal state and params", () => {
    useStore.getState().openInject(true);
    expect(useStore.getState().isInjectOpen).toBe(true);

    useStore.getState().setInjectFamily("Gafgyt");
    expect(useStore.getState().injectFamily).toBe("Gafgyt");

    useStore.getState().setInjectDuration(30);
    expect(useStore.getState().injectDuration).toBe(30);

    useStore.getState().openInject(false);
    expect(useStore.getState().isInjectOpen).toBe(false);
  });

  it("toggles log pause flag", () => {
    expect(useStore.getState().isLogPaused).toBe(false);
    useStore.getState().setIsLogPaused(true);
    expect(useStore.getState().isLogPaused).toBe(true);
    useStore.getState().setIsLogPaused(false);
    expect(useStore.getState().isLogPaused).toBe(false);
  });
});
