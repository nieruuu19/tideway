import type { StreamInfo } from "@/api/types";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  QUALITY_TIER_LABEL,
  QUALITY_TONE,
  qualityTierFor,
} from "@/lib/qualityTiers";
import { cn } from "@/lib/utils";

/**
 * Quality pill in the mini player. Pill text uses TIDAL's own ladder
 * (Low / High / Max, plus Atmos), so settings, the streaming-quality
 * picker, the download menus, and both now-playing surfaces all read
 * the same way. "Low" covers both AAC rungs — TIDAL groups 96 and 320
 * kbps under it.
 *
 * The tooltip on hover keeps the full technical readout — codec +
 * sample rate + bit depth (e.g. "FLAC · 96kHz · 24-bit") — so users
 * who want to confirm they're getting bit-perfect output can still
 * see it without having to dig.
 *
 * Tone is the shared tier palette (`lib/qualityTiers`): neutral for Low,
 * sky for High, gold for Max, violet for Atmos.
 *
 * Renders nothing when we don't have enough info (still loading,
 * codec unknown, etc.) — better to omit than show a placeholder.
 */
export function StreamQualityBadge({
  info,
  className,
}: {
  info: StreamInfo | null | undefined;
  className?: string;
}) {
  if (!info || !info.codec) return null;
  const codec = info.codec.toUpperCase();
  const rate = formatRate(info.sample_rate_hz);
  const depth = info.bit_depth ? `${info.bit_depth}-bit` : null;
  const tier = qualityTierFor(info);
  // The pill carries the tier label; the colour carries the same
  // hi-res / lossless / lossy distinction as the Now Playing pill.
  const tone = QUALITY_TONE[tier].badge;

  // Tooltip stays as a full technical readout — codec + rate + depth.
  // The pill carries the tier label; the tooltip carries the specs
  // for users who care about exact rate and bit depth.
  const fullLabel = [codec, rate ? `${rate}kHz` : null, depth]
    .filter(Boolean)
    .join(" · ");

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span
          className={cn(
            // shrink-0 so a truncating parent (e.g. the
            // artist/metadata row in NowPlaying) can't clip the pill
            // out of existence — the label is already small and the
            // tooltip carries the rest.
            "shrink-0 whitespace-nowrap rounded px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-wider cursor-default",
            tone,
            className,
          )}
        >
          {QUALITY_TIER_LABEL[tier]}
        </span>
      </TooltipTrigger>
      <TooltipContent align="center">{fullLabel}</TooltipContent>
    </Tooltip>
  );
}

function formatRate(hz: number | null): string | null {
  if (!hz || hz <= 0) return null;
  // 44100 → "44.1", 96000 → "96", 192000 → "192". Trim the ".0"
  // off whole-number kHz values.
  const khz = hz / 1000;
  return khz % 1 === 0 ? String(khz) : khz.toFixed(1);
}
