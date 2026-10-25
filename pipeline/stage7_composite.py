"""Stage 7 — Compositing (+ Stage 8 letterbox in the same render pass).

Muxes the assembled background video, the cleaned voiceover audio, and the
burned-in .ass captions, then applies the cinematic letterbox — all in a single
ffmpeg pass to avoid a second generation of x264 re-encoding. Caption vertical
placement (Stage 6) already accounts for the letterbox geometry, so burning the
captions and adding the bars in one chain leaves them correctly inside the
visible picture.
"""
from __future__ import annotations

import logging
from pathlib import Path

from .config import LetterboxConfig, RenderConfig
from .stage8_letterbox import LetterboxGeometry, letterbox_filter
from .utils import find_binary, run

log = logging.getLogger(__name__)


def _escape_for_filter(path: Path) -> str:
    """Escape a Windows path for use inside an ffmpeg filtergraph argument.

    ``C:\\dir\\file.ass`` → ``C\\:/dir/file.ass`` (forward slashes, escaped colon).
    """
    p = str(path).replace("\\", "/")
    # Escape the drive-letter colon (and any other colon) for the filtergraph.
    p = p.replace(":", "\\:")
    return p


def compose_final(background: Path, audio: Path, ass_file: Path,
                  out_path: Path, geo: LetterboxGeometry,
                  render: RenderConfig, letterbox: LetterboxConfig,
                  fonts_dir: Path, progress=None) -> Path:
    """Render the final captioned, letterboxed video with the voiceover audio."""
    background, audio, ass_file = Path(background), Path(audio), Path(ass_file)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if progress:
        progress(0.1, "Compositing & rendering")

    ass_arg = _escape_for_filter(ass_file)
    fonts_arg = _escape_for_filter(Path(fonts_dir))
    box = letterbox_filter(geo, letterbox.pad_colour)

    # Burn captions, then letterbox. ``fontsdir`` makes the bundled font usable
    # without a system install.
    vchain = f"ass='{ass_arg}':fontsdir='{fonts_arg}'"
    if box != "null":
        vchain += f",{box}"

    ffmpeg = find_binary("ffmpeg")
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(background),
        "-i", str(audio),
        "-filter_complex", f"[0:v]{vchain}[v]",
        "-map", "[v]", "-map", "1:a",
        "-c:v", render.video_codec, "-crf", str(render.crf),
        "-preset", render.preset, "-pix_fmt", "yuv420p",
        "-c:a", render.audio_codec, "-b:a", render.audio_bitrate,
        "-shortest", "-movflags", "+faststart",
        str(out_path),
    ]
    run(cmd, desc="final render (composite + captions + letterbox)")

    if progress:
        progress(1.0, "Final render complete")
    log.info("Stage 7/8 complete -> %s", out_path)
    return out_path
