"""PyAV decoder wrapper — yields PCM in the source's native format.

For bit-perfect output, the decoder auto-detects the source's
sample format and configures its `AudioResampler` to emit packed
(interleaved) samples in the SAME format — no format conversion,
no sample-rate conversion. The resampler is only there because
PyAV's decoded frames come out in planar layouts
(`s16p`/`s32p`/`fltp`) and sounddevice needs packed; interleaving
is lossless.

Format mapping:
  FLAC 16-bit → `s16`  → numpy int16   → sounddevice `int16`
  FLAC 24-bit → `s32`  → numpy int32   → sounddevice `int32`
  AAC / MP3   → `fltp` → numpy float32 → sounddevice `float32`

Anything unrecognized falls through to float32, which sounddevice
always supports.
"""
from __future__ import annotations

import io
import logging
import sys
import threading
import time
from dataclasses import dataclass
from typing import Iterator, Optional, Union

import av  # type: ignore
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class CodecInfo:
    codec: str
    sample_rate: int
    channels: int
    source_format: Optional[str]
    duration_seconds: Optional[float]
    bit_depth: Optional[int]


# Source format name -> (packed resampler format, numpy dtype, sd dtype, bit_depth)
_FORMAT_MAP = {
    "s16":  ("s16", np.int16,   "int16",   16),
    "s16p": ("s16", np.int16,   "int16",   16),
    "s32":  ("s32", np.int32,   "int32",   24),
    "s32p": ("s32", np.int32,   "int32",   24),
    "flt":  ("flt", np.float32, "float32", None),
    "fltp": ("flt", np.float32, "float32", None),
    "dbl":  ("flt", np.float32, "float32", None),
    "dblp": ("flt", np.float32, "float32", None),
}

# 10 ** (-1/20). Applied to the decoder's float32 output when the
# decoder is doing internal sample-rate conversion (source rate
# differs from target rate). Reconstruction across an SRC stage can
# produce intersample peaks slightly above 0 dBFS on loud-mastered
# tracks; a 1 dB attenuation in front of the resampler keeps those
# peaks under full scale without the heavy hand of full loudness
# normalization. When source rate equals target rate, no resampling
# happens and headroom stays at 1.0 (bit-perfect).
_RESAMPLE_HEADROOM = 0.8912509381337456


class Decoder:
    """One `Decoder` = one track.

    `source` may be a filesystem path (for local files) or any
    file-like with read/seek (for streamed DASH via `SegmentReader`).

    `next_pcm()` belongs to the decode thread; `request_seek()` may be
    called from any thread — it's just a flag the decode loop reads
    between frames. `close()` may also come from any thread (teardown,
    preload drop, the cross-rate bridge) and is serialized against
    `next_pcm` by `_io_lock`: freeing the PyAV container while the
    decode thread is inside libavformat was an access violation that
    killed the backend (crash dump from the `pcm-preload` thread in
    avformat-63, during a fast track change that dropped a preload).
    """

    # How long close() waits for an in-flight decode to leave libav.
    # Bounded because the decode thread can be parked in a network read;
    # on timeout we leave the container to the thread/GC rather than
    # block the caller. cancel_source() before close() makes that rare.
    _CLOSE_LOCK_WAIT_S = 1.0

    def __init__(self, source: Union[str, io.IOBase], *, surround: bool = False):
        self._io_lock = threading.Lock()
        self._closed = False
        self._source = source
        # libav's default probesize (5 MB) and analyzeduration
        # (5 seconds of audio) cause av.open to read several DASH
        # segments before it's satisfied it knows what the stream
        # is — adding seconds to every play on hi-res. We already
        # know from Tidal's manifest that it's FLAC (or AAC) in
        # fMP4; the moov atom in the init segment plus ~200ms of
        # audio is plenty. If these values ever reject a legitimate
        # stream we fall back to the libav defaults before giving up.
        open_opts = {
            "probesize": "131072",
            "analyzeduration": "200000",
        }
        # Tidal serves fragmented MP4 at every quality tier, so we
        # can short-circuit libav's format-detection phase by naming
        # the container explicitly. Only applies to the streaming
        # path — local files go through default sniffing so arbitrary
        # downloaded formats (flac, mp3, m4a) still open.
        open_fmt = "mp4" if not isinstance(source, str) else None
        t0 = time.monotonic()
        try:
            self._container = av.open(
                source, format=open_fmt, options=open_opts
            )
        except Exception as first_exc:
            # Reopen with defaults as a safety net. The source must
            # be re-seekable for this to work; both our SegmentReader
            # and local filesystem paths support that.
            print(
                f"[perf] decoder av.open with probesize={open_opts['probesize']} "
                f"failed ({first_exc!r}); retrying with libav defaults",
                file=sys.stderr,
                flush=True,
            )
            try:
                if hasattr(source, "seek"):
                    source.seek(0)  # type: ignore[union-attr]
            except Exception:
                pass
            self._container = av.open(source)
        t_open = time.monotonic()
        streams = [s for s in self._container.streams if s.type == "audio"]
        if not streams:
            self._container.close()
            raise RuntimeError("no audio stream in source")
        self._stream = streams[0]
        cc = self._stream.codec_context
        self._sample_rate = int(cc.sample_rate)
        print(
            f"[perf] decoder av.open={((t_open - t0) * 1000.0):.0f}ms "
            f"codec_setup={((time.monotonic() - t_open) * 1000.0):.0f}ms",
            file=sys.stderr,
            flush=True,
        )
        channels = getattr(cc, "channels", None)
        if channels is None:
            layout = getattr(cc, "layout", None) or getattr(cc, "channel_layout", None)
            channels = getattr(layout, "nb_channels", None) or 2
        self._source_channels = int(channels)
        # Multichannel sources downmix to stereo by default: a 5.1 file
        # played on a stereo device should still be audible. `surround`
        # opts into passing the native layout through instead, which is
        # what Atmos (E-AC-3) needs — its whole point is the 5.1/7.1
        # bed. The player only sets it after confirming the output
        # device can take the channels; otherwise it calls
        # set_target_channels(2) and this collapses to the default.
        self._output_channels = (
            self._source_channels
            if surround and self._source_channels in (6, 8)
            else (2 if self._source_channels > 2 else self._source_channels)
        )

        src_fmt = getattr(cc, "format", None)
        src_fmt_name = getattr(src_fmt, "name", None)
        out_fmt, dtype, sd_dtype, bit_depth = _FORMAT_MAP.get(
            src_fmt_name or "", ("flt", np.float32, "float32", None)
        )
        self._source_format = src_fmt_name
        self._source_format_packed = out_fmt
        self._source_dtype = dtype
        self._source_sd_dtype = sd_dtype
        self._bit_depth = bit_depth
        self._target_rate: Optional[int] = None
        self._output_format = out_fmt
        self._output_dtype = dtype
        self._sd_dtype = sd_dtype
        self._headroom = (
            _RESAMPLE_HEADROOM if self._source_channels > self._output_channels else 1.0
        )
        self._resampler = self._make_resampler()
        self._iter: Optional[Iterator[av.AudioFrame]] = None
        self._done = False
        self._pending_seek_s: Optional[float] = None

    # --- public API -------------------------------------------------

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def output_sample_rate(self) -> int:
        return self._target_rate if self._target_rate else self._sample_rate

    @property
    def source_channels(self) -> int:
        return self._source_channels

    @property
    def channels(self) -> int:
        return self._output_channels

    def set_target_channels(self, target_channels: int) -> None:
        target_channels = max(1, int(target_channels))
        if target_channels == self._output_channels:
            return
        self._output_channels = target_channels
        needs_headroom = (
            self._target_rate is not None
            or self._source_channels > self._output_channels
        )
        self._headroom = _RESAMPLE_HEADROOM if needs_headroom else 1.0
        self._resampler = self._make_resampler()

    @property
    def output_dtype(self) -> type:
        return self._output_dtype

    @property
    def sounddevice_dtype(self) -> str:
        return self._sd_dtype

    @property
    def is_resampling_internally(self) -> bool:
        return self._target_rate is not None

    def set_target_rate(self, rate: int, *, float_output: bool = False) -> None:
        """Reconfigure the decoder's output rate.

        If `rate` equals the source sample rate we stay in bit-perfect
        passthrough: source format, source dtype, no attenuation. If
        it differs we flip to float32 at the target rate and apply a
        small input attenuation so the resampler has headroom for
        intersample-peak overshoot, the only thing that audibly
        clipped under the previous "let the OS resample" behavior.

        `float_output` asks for float32 at the source rate too, still
        without attenuation: 16- and 24-bit samples convert to float32
        exactly, so the values are the ones PortAudio would have
        produced converting integers for a float32 OS mixer. The player
        sets it where that mixer is the destination anyway (see
        PCMPlayer._float_output_for).

        Idempotent. Calling with the rate and format we're already
        configured for is a no-op and does NOT rebuild the resampler —
        important for the gapless-bridge path where the preload's
        decoder thread is mid-decode and a resampler rebuild would drop
        any samples buffered inside libav."""
        rate = int(rate)
        wants_passthrough = rate <= 0 or rate == self._sample_rate
        downmixing = self._source_channels > self._output_channels
        if wants_passthrough:
            if float_output:
                fmt, dtype, sd_dtype = "flt", np.float32, "float32"
            else:
                fmt = self._source_format_packed
                dtype = self._source_dtype
                sd_dtype = self._source_sd_dtype
            if self._target_rate is None and self._output_format == fmt:
                return
            self._target_rate = None
            self._output_format = fmt
            self._output_dtype = dtype
            self._sd_dtype = sd_dtype
            self._headroom = _RESAMPLE_HEADROOM if downmixing else 1.0
        else:
            if self._target_rate == rate:
                return
            self._target_rate = rate
            self._output_format = "flt"
            self._output_dtype = np.float32
            self._sd_dtype = "float32"
            self._headroom = _RESAMPLE_HEADROOM
        self._resampler = self._make_resampler()

    def codec_info(self) -> CodecInfo:
        return CodecInfo(
            codec=self._stream.codec_context.name,
            sample_rate=self._sample_rate,
            channels=self._source_channels,
            source_format=self._source_format,
            duration_seconds=self._duration_seconds(),
            bit_depth=self._bit_depth,
        )

    def request_seek(self, position_seconds: float) -> None:
        if self._closed:
            return
        self._pending_seek_s = float(max(0.0, position_seconds))

    def next_pcm(self) -> Optional[np.ndarray]:
        if self._closed:
            return None
        with self._io_lock:
            # Re-check under the lock: close() sets the flag before it
            # waits for us, so a decode that lost the race must not
            # touch a container that is about to be freed.
            if self._closed:
                return None
            return self._next_pcm_locked()

    def _next_pcm_locked(self) -> Optional[np.ndarray]:
        if self._done and self._pending_seek_s is None:
            return None
        if self._pending_seek_s is not None:
            target_s = self._pending_seek_s
            self._pending_seek_s = None
            try:
                self._container.seek(
                    int(target_s / float(self._stream.time_base)),
                    stream=self._stream,
                    any_frame=False,
                )
            except Exception:
                log.exception("container.seek failed for target_s=%s", target_s)
            try:
                self._stream.codec_context.flush_buffers()
            except Exception:
                pass
            self._resampler = self._make_resampler()
            self._iter = None
            self._done = False

        if self._iter is None:
            self._iter = self._container.decode(self._stream)
        while True:
            try:
                frame = next(self._iter)
            except StopIteration:
                tail = self._resampler.resample(None)
                self._done = True
                if tail:
                    return self._emit(tail)
                return None
            resampled = self._resampler.resample(frame)
            if not resampled:
                continue
            return self._emit(resampled)

    def _emit(self, frames: list) -> np.ndarray:
        arr = _frames_to_packed(frames, self._output_channels, self._output_dtype)
        if self._headroom != 1.0:
            arr *= self._headroom
        return arr

    def cancel_source(self) -> None:
        if hasattr(self._source, "close"):
            try:
                self._source.close()  # type: ignore[union-attr]
            except Exception:
                pass

    def close(self) -> None:
        # Set the flag first (it's what next_pcm/_next_pcm_locked check)
        # so the decode thread stops re-entering libav either way.
        self._closed = True
        acquired = self._io_lock.acquire(timeout=self._CLOSE_LOCK_WAIT_S)
        if not acquired:
            log.warning(
                "decoder close skipped PyAV teardown: decode thread still "
                "inside libav; leaving the container to the thread/GC"
            )
        try:
            if acquired:
                try:
                    self._container.close()
                except Exception:
                    pass
        finally:
            if acquired:
                self._io_lock.release()
        self.cancel_source()

    # --- internals --------------------------------------------------

    def _make_resampler(self) -> av.AudioResampler:
        layout_map = {
            1: "mono",
            2: "stereo",
            6: "5.1",
            8: "7.1",
        }
        layout = layout_map.get(self._output_channels, "stereo")
        return av.AudioResampler(
            format=self._output_format,
            layout=layout,
            rate=self._target_rate if self._target_rate else self._sample_rate,
        )

    def _duration_seconds(self) -> Optional[float]:
        dur = self._container.duration
        if dur is None or dur <= 0:
            return None
        return float(dur) / 1_000_000.0


def _frames_to_packed(frames: list, channels: int, dtype: type) -> np.ndarray:
    """Stack PyAV AudioFrames (packed) into one (N, channels) array."""
    chunks: list[np.ndarray] = []
    for f in frames:
        arr = f.to_ndarray()
        if arr.ndim == 2 and arr.shape[0] == 1:
            arr = arr.reshape(-1, channels)
        elif arr.ndim == 2 and arr.shape[0] == channels:
            arr = arr.T
        chunks.append(arr)
    if not chunks:
        return np.zeros((0, channels), dtype=dtype)
    return np.concatenate(chunks, axis=0).astype(dtype, copy=False)
