"""Stage 1 — Audio preparation (pass-through).

Voice enhancement (resemble-enhance) has been intentionally removed: we keep the
**original recorded voice** untouched. This stage only normalises the container
— it extracts the audio stream from the input (audio file *or* video like
mp4/mkv) into a wav for the rest of the pipeline to consume, preserving the
original sample rate and channel layout. No denoising, EQ, or any processing is
applied. This stage does not use the GPU.
"""
from __future__ import annotations

import logging
from pathlib import Path

from .utils import PipelineError, run_ffmpeg

log = logging.getLogger(__name__)


def prepare_audio(input_audio: Path, out_path: Path, progress=None) -> Path:
    """Decode the original audio stream to a wav without any enhancement.

    Preserves the source sample rate and channel count (no resampling, no
    downmix) so the voice is identical to the recording — just in wav form.
    Works for both audio files and video containers (the audio track is
    extracted; any video stream is dropped).
    """
    input_audio = Path(input_audio)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if progress:
        progress(0.2, "Extracting original audio")

    try:
        run_ffmpeg(
            ["-i", str(input_audio), "-vn", "-c:a", "pcm_s16le", str(out_path)],
            desc="extract original audio",
        )
    except PipelineError as exc:
        raise PipelineError(
            f"Could not read an audio track from {input_audio.name}. If this is "
            f"a video file, make sure it actually contains an audio stream.\n{exc}"
        ) from exc

    if progress:
        progress(1.0, "Audio ready (original, unprocessed)")
    log.info("Stage 1 (prepare, no enhancement) -> %s", out_path)
    return out_path
