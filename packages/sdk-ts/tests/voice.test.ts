import { describe, it, expect } from "vitest";
import { DunetraceRun } from "../src/run.js";
import type { AgentEvent } from "../src/models.js";

function makeRun() {
  const emitted: AgentEvent[] = [];
  const emitter = { _emit: (e: AgentEvent) => emitted.push(e) };
  const run = new DunetraceRun("voice-agent", "abc12345", emitter);
  return { run, emitted };
}

function last(emitted: AgentEvent[]): AgentEvent {
  return emitted[emitted.length - 1];
}

describe("DunetraceRun — voice hooks", () => {
  it("transcriptionReceived advances the step and carries fields", () => {
    const { run, emitted } = makeRun();
    run.transcriptionReceived("where is my order", { confidence: 0.9, latencyMs: 120, audioSeconds: 1.5 });
    expect(run.currentStep()).toBe(1);
    const e = last(emitted);
    expect(e.event_type).toBe("transcription.received");
    expect(e.payload).toMatchObject({ text: "where is my order", confidence: 0.9, latency_ms: 120, audio_seconds: 1.5 });
  });

  it("transcriptionReceived omits audio_seconds when not given", () => {
    const { run, emitted } = makeRun();
    run.transcriptionReceived("hi");
    expect(last(emitted).payload).not.toHaveProperty("audio_seconds");
  });

  it("ttsGenerated does NOT advance the step and carries provider metadata", () => {
    const { run, emitted } = makeRun();
    run.ttsGenerated("your order ships today", { latencyMs: 90, voiceId: "rachel", model: "eleven_turbo_v2" });
    expect(run.currentStep()).toBe(0);
    const e = last(emitted);
    expect(e.event_type).toBe("tts.generated");
    expect(e.payload).toMatchObject({ text: "your order ships today", latency_ms: 90, voice_id: "rachel", model: "eleven_turbo_v2", truncated: false });
  });

  it("voiceActivityDetected does NOT advance the step", () => {
    const { run, emitted } = makeRun();
    run.voiceActivityDetected("silence", 800);
    expect(run.currentStep()).toBe(0);
    expect(last(emitted)).toMatchObject({ event_type: "voice_activity.detected", payload: { type: "silence", duration_ms: 800 } });
  });

  it("voiceActivityDetected rejects an unknown type", () => {
    const { run } = makeRun();
    // @ts-expect-error deliberately invalid
    expect(() => run.voiceActivityDetected("mumble")).toThrow(/type must be one of/);
  });

  it("turnTaking does NOT advance the step and validates the action", () => {
    const { run, emitted } = makeRun();
    run.turnTaking("agent_speaking", true, false);
    expect(run.currentStep()).toBe(0);
    expect(last(emitted)).toMatchObject({ event_type: "turn_taking.changed", payload: { action: "agent_speaking", from_agent: true, to_user: false } });
    // @ts-expect-error deliberately invalid
    expect(() => run.turnTaking("shouting")).toThrow(/action must be one of/);
  });

  it("recordingMetadata does NOT advance the step and includes only set fields", () => {
    const { run, emitted } = makeRun();
    run.recordingMetadata("https://audio.example/call.wav", { durationSeconds: 42, format: "wav" });
    expect(run.currentStep()).toBe(0);
    const e = last(emitted);
    expect(e.event_type).toBe("recording.available");
    expect(e.payload).toMatchObject({ url: "https://audio.example/call.wav", duration_seconds: 42, format: "wav" });
    expect(e.payload).not.toHaveProperty("storage_provider");
  });

  it("a full voice turn is ~one step (transcription advances, the rest annotate)", () => {
    const { run } = makeRun();
    run.transcriptionReceived("hello");
    run.voiceActivityDetected("speech_end");
    run.ttsGenerated("hi back");
    run.turnTaking("agent_speaking");
    expect(run.currentStep()).toBe(1);
  });
});
