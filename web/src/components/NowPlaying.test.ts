import { describe, expect, it } from "vitest";
import type { StreamInfo } from "@/api/types";
import { streamInfoFullLabel, streamingQualityLabel } from "./NowPlaying";

function info(over: Partial<StreamInfo>): StreamInfo {
  return {
    source: "stream",
    codec: null,
    bit_depth: null,
    sample_rate_hz: null,
    audio_quality: null,
    audio_mode: null,
    ...over,
  };
}

describe("streamInfoFullLabel", () => {
  it("labels a real Atmos asset from the Atmos resolver", () => {
    expect(
      streamInfoFullLabel(
        info({
          codec: "eac3",
          sample_rate_hz: 48000,
          audio_quality: "DOLBY_ATMOS",
          audio_mode: "DOLBY_ATMOS",
        }),
      ),
    ).toBe("Atmos · 48kHz");
  });

  it("labels a local E-AC-3/AC-4 download as Atmos", () => {
    expect(
      streamInfoFullLabel(
        info({ source: "local", codec: "eac3", sample_rate_hz: 48000 }),
      ),
    ).toBe("Atmos · 48kHz");
    expect(
      streamInfoFullLabel(
        info({ source: "local", codec: "ac4", sample_rate_hz: 48000 }),
      ),
    ).toBe("Atmos · 48kHz");
  });

  it("does not claim Atmos for a lossy fallback that echoes the mode", () => {
    expect(
      streamInfoFullLabel(
        info({
          codec: "aac",
          sample_rate_hz: 44100,
          audio_quality: "LOW",
          audio_mode: "DOLBY_ATMOS",
        }),
      ),
    ).toBe("44.1kHz");
  });

  it("keeps depth for lossless streams", () => {
    expect(
      streamInfoFullLabel(
        info({
          codec: "flac",
          sample_rate_hz: 44100,
          bit_depth: 16,
          audio_quality: "LOSSLESS",
          audio_mode: "STEREO",
        }),
      ),
    ).toBe("44.1kHz · 16-bit");
  });

  it("leaves Atmos naming to the Dolby badge once the backend classified", () => {
    // Rendered as objects, and audible only as the bed: same stream
    // readout, and the badge beside the pill says which.
    for (const immersive of ["atmos", "dolby_audio"] as const) {
      expect(
        streamInfoFullLabel(
          info({
            source: "local",
            codec: "eac3",
            sample_rate_hz: 48000,
            audio_mode: "DOLBY_ATMOS",
            immersive,
          }),
        ),
      ).toBe("48kHz");
    }
  });

  it("returns null without usable info", () => {
    expect(streamInfoFullLabel(null)).toBeNull();
    expect(streamInfoFullLabel(info({}))).toBeNull();
  });
});

describe("streamingQualityLabel", () => {
  it("files both AAC rungs under Low, like TIDAL's dialog", () => {
    expect(streamingQualityLabel("low_96k")).toBe("Low");
    expect(streamingQualityLabel("low_320k")).toBe("Low");
  });

  it("keeps the lossless and Atmos rows distinct", () => {
    expect(streamingQualityLabel("high_lossless")).toBe("High");
    expect(streamingQualityLabel("hi_res_lossless")).toBe("Max");
    expect(streamingQualityLabel("atmos")).toBe("Dolby Atmos");
  });
});
