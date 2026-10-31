/**
 * One palette for every place the app names an audio-quality tier.
 *
 * The Now Playing picker established it, following TIDAL's own ladder:
 *   Low   → neutral  (AAC at 96 or 320 kbps — lossy, no special colour)
 *   High  → sky      (FLAC 16-bit / 44.1 kHz)
 *   Max   → gold     (FLAC up to 24-bit / 192 kHz)
 *   Atmos → violet   (E-AC-3 JOC — a format, not a rung on the ladder)
 *
 * Album badges, the mini-player pill, the download menus and the Atmos
 * marker all read from here so one tier never shows up in two colours.
 * Pinned Tailwind hues rather than theme tokens so they hold on the
 * dark palette the app ships and on the light one.
 */
export type QualityTier = "low" | "high" | "max" | "atmos";

export const QUALITY_TONE: Record<
  QualityTier,
  {
    /** Foreground colour for a label or tag. */
    text: string;
    /** Tinted chip, no interaction states. */
    badge: string;
    /** Tinted chip for something clickable (the quality picker's pill). */
    pill: string;
  }
> = {
  low: {
    text: "text-neutral-400",
    badge: "bg-neutral-500/15 text-neutral-400",
    pill: "bg-neutral-500/15 text-neutral-400 hover:bg-neutral-500/25",
  },
  high: {
    text: "text-sky-500",
    badge: "bg-sky-500/15 text-sky-500",
    pill: "bg-sky-500/15 text-sky-500 hover:bg-sky-500/25",
  },
  max: {
    text: "text-amber-500",
    badge: "bg-amber-500/15 text-amber-500",
    pill: "bg-amber-500/15 text-amber-500 hover:bg-amber-500/25",
  },
  atmos: {
    text: "text-violet-400",
    badge: "bg-violet-500/15 text-violet-400",
    pill: "bg-violet-500/15 text-violet-400 hover:bg-violet-500/25",
  },
};

/** Tier label TIDAL's clients use, keyed by our internal tier. */
export const QUALITY_TIER_LABEL: Record<QualityTier, string> = {
  low: "Low",
  high: "High",
  max: "Max",
  atmos: "Dolby Atmos",
};

/**
 * Map a StreamInfo to the shared tier. Tidal streams carry the tier in
 * `audio_quality` directly, so prefer that. Local files don't have it
 * (Tidal isn't involved at all), so we fall back to deriving from
 * codec + sample rate / bit depth.
 */
export function qualityTierFor(info: {
  audio_quality?: string | null;
  codec?: string | null;
  bit_depth?: number | null;
  sample_rate_hz?: number | null;
}): QualityTier {
  const aq = (info.audio_quality || "").toUpperCase();
  if (aq === "LOW" || aq === "HIGH") return "low";
  if (aq === "LOSSLESS") return "high";
  if (aq === "HI_RES" || aq === "HI_RES_LOSSLESS") return "max";
  if (aq === "DOLBY_ATMOS") return "atmos";

  const codec = (info.codec || "").toLowerCase();
  if (codec === "eac3" || codec === "ac4") return "atmos";
  if (codec === "flac" || codec === "alac") {
    const isHiRes =
      (info.bit_depth != null && info.bit_depth >= 24) ||
      (info.sample_rate_hz != null && info.sample_rate_hz > 48000);
    return isHiRes ? "max" : "high";
  }
  // Lossy local file (rare). "low" is TIDAL's bucket for every AAC
  // rung; we have no way to distinguish 96 kbps from 320 kbps sources
  // without Tidal's tier string.
  return "low";
}
