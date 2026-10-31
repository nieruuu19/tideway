import { QUALITY_TONE } from "@/lib/qualityTiers";
import { cn } from "@/lib/utils";

/**
 * TIDAL's Dolby Atmos marker, rendered next to a track or album title.
 *
 * Mirrors the "Saved" pill's metrics so the two read as one badge row;
 * violet comes from the shared tier palette, matching the Atmos pill
 * in Now Playing.
 */
export function AtmosBadge({ className }: { className?: string }) {
  return (
    <span
      title="Dolby Atmos mix — TIDAL serves this release as E-AC-3 JOC"
      className={cn(
        "shrink-0 rounded-sm px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-wider",
        QUALITY_TONE.atmos.badge,
        className,
      )}
    >
      Atmos
    </span>
  );
}
