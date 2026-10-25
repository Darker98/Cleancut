"""Stage 5 — Background clip assembly.

Builds one continuous, silent background video track matching the full audio
duration. Segments are ≤ ``segment_seconds`` or shorter when a ``[BREAK]`` falls
earlier. Each segment is filled by the next clip from the folder; clips that are
too short are extended via a reverse-loop (ping-pong: clip + reversed + clip …)
so the seam is invisible, then trimmed to the exact target length.
"""
from __future__ import annotations

import logging
import math
import random
from pathlib import Path

from .config import BackgroundConfig, RenderConfig
from .utils import PipelineError, probe_duration, run_ffmpeg

log = logging.getLogger(__name__)


def _scale_pad_filter(w: int, h: int) -> str:
    """Scale to fit WxH preserving aspect, then pad to exactly WxH (centered)."""
    return (
        f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
        f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1"
    )


def _normalize_clip(src: Path, dst: Path, render: RenderConfig,
                    duration: float | None, reverse: bool = False) -> Path:
    """Normalise a clip to render geometry/fps, silent, optionally reversed.

    If ``duration`` is given the output is trimmed to it.
    """
    vf = _scale_pad_filter(render.width, render.height)
    vf += f",fps={render.fps}"
    if reverse:
        vf += ",reverse"
    args = ["-i", str(src), "-an", "-vf", vf]
    if duration is not None:
        args += ["-t", f"{duration:.3f}"]
    args += [
        "-c:v", render.video_codec, "-crf", str(render.crf),
        "-preset", render.preset, "-pix_fmt", "yuv420p", str(dst),
    ]
    run_ffmpeg(args, desc=f"normalize clip {src.name}")
    return dst


def _build_segment(clip: Path, target: float, dst: Path, render: RenderConfig,
                   work: Path, seg_idx: int) -> Path:
    """Produce a clip exactly ``target`` seconds long at render geometry."""
    clip_dur = probe_duration(clip)

    if clip_dur >= target:
        # Long enough — just trim.
        return _normalize_clip(clip, dst, render, duration=target)

    # Too short → ping-pong reverse-loop until we cover the target, then trim.
    fwd = work / f"seg{seg_idx}_fwd.mp4"
    rev = work / f"seg{seg_idx}_rev.mp4"
    _normalize_clip(clip, fwd, render, duration=None, reverse=False)
    _normalize_clip(clip, rev, render, duration=None, reverse=True)

    # Number of half-cycles needed (each ~clip_dur long), alternating fwd/rev.
    n_parts = max(2, math.ceil(target / clip_dur))
    list_file = work / f"seg{seg_idx}_concat.txt"
    lines = []
    for i in range(n_parts):
        part = fwd if i % 2 == 0 else rev
        lines.append(f"file '{part.as_posix()}'")
    list_file.write_text("\n".join(lines), encoding="utf-8")

    looped = work / f"seg{seg_idx}_looped.mp4"
    run_ffmpeg(
        ["-f", "concat", "-safe", "0", "-i", str(list_file),
         "-c", "copy", str(looped)],
        desc=f"ping-pong concat seg {seg_idx}",
    )
    # Trim the looped result to the exact target.
    run_ffmpeg(
        ["-i", str(looped), "-t", f"{target:.3f}",
         "-c:v", render.video_codec, "-crf", str(render.crf),
         "-preset", render.preset, "-pix_fmt", "yuv420p", str(dst)],
        desc=f"trim looped seg {seg_idx}",
    )
    return dst


def assemble_background(segments: list[tuple[float, float]], clips: list[Path],
                        out_path: Path, bg_cfg: BackgroundConfig,
                        render: RenderConfig, work: Path, progress=None) -> Path:
    """Assemble the background track for the given segment list."""
    if not segments:
        raise PipelineError("No background segments to assemble (empty timeline).")
    if not clips:
        raise PipelineError("No background clips provided.")

    work.mkdir(parents=True, exist_ok=True)

    # Decide clip order.
    order = list(range(len(clips)))
    if bg_cfg.order == "random":
        rng = random.Random(bg_cfg.random_seed)
        rng.shuffle(order)

    targets = [end - start for (start, end) in segments]
    n = len(segments)

    # Decide the crossfade duration. With a 2s dissolve, each segment (except the
    # last) is built `t` seconds longer so the overlap is consumed and the final
    # track still matches the audio duration exactly. Clamp `t` so it never
    # exceeds the shortest segment.
    t = max(0.0, float(bg_cfg.crossfade_seconds))
    if n > 1 and t > 0:
        t = min(t, max(0.0, min(targets) * 0.9))
    do_xfade = n > 1 and t > 0.05

    seg_files: list[Path] = []
    for i, (start, end) in enumerate(segments):
        target = targets[i]
        # Non-final segments get +t to feed the crossfade overlap.
        build_dur = target + t if (do_xfade and i < n - 1) else target
        clip = clips[order[i % len(order)]]
        if progress:
            progress(i / n,
                     f"Segment {i + 1}/{n} ({target:.1f}s) ← {clip.name}")
        dst = work / f"segment_{i:03d}.mp4"
        _build_segment(clip, build_dur, dst, render, work, i)
        seg_files.append(dst)

    if not do_xfade:
        # Hard cuts — identical codec/geometry, so a stream copy concat is safe.
        list_file = work / "background_concat.txt"
        list_file.write_text(
            "\n".join(f"file '{p.as_posix()}'" for p in seg_files), encoding="utf-8"
        )
        run_ffmpeg(
            ["-f", "concat", "-safe", "0", "-i", str(list_file),
             "-c", "copy", str(out_path)],
            desc="concat background segments",
        )
    else:
        # Chain xfade dissolves. The k-th xfade (adding segment k) starts at
        # offset = sum(targets[0:k]); the running result length after it is
        # sum(targets[0:k+1]) + t, and the final length collapses to sum(targets)
        # == the audio duration.
        inputs: list[str] = []
        for p in seg_files:
            inputs += ["-i", str(p)]
        filt = []
        prev = "[0:v]"
        cum = 0.0
        for k in range(1, n):
            cum += targets[k - 1]
            out_label = "[vout]" if k == n - 1 else f"[vx{k}]"
            filt.append(
                f"{prev}[{k}:v]xfade=transition=fade:duration={t:.3f}:"
                f"offset={cum:.3f}{out_label}"
            )
            prev = out_label
        if progress:
            progress(0.95, f"Crossfading {n} clips ({t:.1f}s dissolve)")
        run_ffmpeg(
            [*inputs, "-filter_complex", ";".join(filt), "-map", "[vout]",
             "-c:v", render.video_codec, "-crf", str(render.crf),
             "-preset", render.preset, "-pix_fmt", "yuv420p", str(out_path)],
            desc="crossfade background segments",
        )

    if progress:
        progress(1.0, "Background track assembled")
    log.info("Stage 5 complete: %d segments (%s) -> %s", n,
             f"{t:.1f}s crossfade" if do_xfade else "hard cuts", out_path)
    return out_path
