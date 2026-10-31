import { useRef, useState } from "react";

import { formatDuration } from "@/lib/utils";

/**
 * Seek control shared by the now-playing bar and the full-screen
 * player.
 *
 * A plain controlled range input fires `change` for every value the
 * pointer crosses, and each of those used to become a POST to
 * `/api/player/seek`. The backend handles those one at a time under its
 * pipeline lock — a DASH seek rebuilds the decoder and a native Atmos
 * seek round-trips to MediaPlayer — so a single drag queued dozens of
 * them and kept applying after the mouse was released: playback then
 * walked backwards through every position the drag had passed over.
 *
 * So a pointer gesture only previews here; the seek is sent once, on
 * release. Keyboard and assistive-tech changes have no gesture to wait
 * for and seek immediately, one per key press.
 */
export function SeekBar({
  currentTime,
  duration,
  onSeek,
  variant = "bar",
}: {
  currentTime: number;
  duration: number;
  onSeek: (seconds: number) => void;
  variant?: "bar" | "overlay";
}) {
  const [scrub, setScrub] = useState<number | null>(null);
  const scrubbing = useRef(false);

  const shown = scrub ?? Math.min(currentTime, duration || 0);
  const pct = duration > 0 ? Math.min(100, (shown / duration) * 100) : 0;

  const release = (seek: number | null) => {
    if (!scrubbing.current) return;
    scrubbing.current = false;
    setScrub(null);
    if (seek !== null) onSeek(seek);
  };

  return (
    <div
      className={
        variant === "overlay"
          ? "flex w-full max-w-xl items-center gap-2 text-[11px] text-foreground/70"
          : "group flex w-full max-w-xl items-center gap-2 text-[11px] text-muted-foreground"
      }
    >
      <span className="w-10 text-right tabular-nums">
        {formatDuration(shown)}
      </span>
      <input
        type="range"
        min={0}
        max={Math.max(duration, 1)}
        step={0.1}
        value={shown}
        aria-label="Seek"
        className={
          variant === "overlay"
            ? "h-1 flex-1 cursor-pointer appearance-none rounded-full accent-primary"
            : "h-1 flex-1 cursor-pointer appearance-none rounded-full bg-secondary accent-primary transition-[height] duration-150 ease-out group-hover:h-1.5"
        }
        style={{
          background:
            variant === "overlay"
              ? `linear-gradient(to right, white ${pct}%, rgba(255,255,255,0.2) ${pct}%)`
              : `linear-gradient(to right, hsl(var(--primary)) ${pct}%, hsl(var(--secondary)) ${pct}%)`,
        }}
        onPointerDown={(e) => {
          scrubbing.current = true;
          setScrub(Number(e.currentTarget.value));
          try {
            e.currentTarget.setPointerCapture(e.pointerId);
          } catch {
            // No pointer capture (older engines): release still fires
            // on the element itself, and a lost gesture is discarded
            // by the pointercancel/lostpointercapture handlers below.
          }
        }}
        onInput={(e) => {
          // `input`, not `change`: Chromium fires one trailing `change`
          // after the gesture ends, by which point the value has been
          // reset to the player's position — handling it would send a
          // second seek that undoes the one just committed on release.
          const value = Number(e.currentTarget.value);
          if (scrubbing.current) setScrub(value);
          else onSeek(value);
        }}
        onPointerUp={(e) => release(Number(e.currentTarget.value))}
        onPointerCancel={() => release(null)}
        onLostPointerCapture={(e) => release(Number(e.currentTarget.value))}
      />
      <span className="w-10 tabular-nums">{formatDuration(duration)}</span>
    </div>
  );
}
