"""Stage 4 — Pause / dead-air trimming via auto-editor.

Runs on the (stutter-cut) audio. auto-editor detects silent regions and removes
them, keeping a configurable margin around speech so cuts don't sound clipped.
Because this changes the timeline, the caller re-transcribes the output to get
authoritative word timestamps for alignment and captions.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

from .config import PauseConfig
from .utils import PipelineError, find_binary, run

log = logging.getLogger(__name__)


def trim_pauses(audio_path: Path, out_path: Path, cfg: PauseConfig,
                progress=None) -> Path:
    """Trim long silences from ``audio_path``; write result to ``out_path``."""
    audio_path = Path(audio_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resolve the auto-editor executable (installed as a console script in venv).
    try:
        exe = find_binary("auto-editor")
    except PipelineError:
        exe = shutil.which("auto-editor.exe")
        if not exe:
            raise PipelineError(
                "auto-editor not found. Install with `pip install auto-editor` "
                "inside the venv."
            )

    if progress:
        progress(0.2, "Trimming silences")

    # auto-editor 29.x: --edit "audio:threshold=N%" sets the loudness gate;
    # silent sections are cut by default (--when-silent defaults to "cut").
    # --margin keeps padding around kept speech, so silences shorter than
    # ~2×margin are bridged (preserved). We map the "min pause to cut" control
    # to margin = min_pause/2: pauses shorter than min_pause stay intact, longer
    # ones are trimmed down to roughly min_pause of retained air.
    # The output format is inferred from the .wav extension (no --export flag).
    threshold_pct = f"{cfg.silence_threshold * 100:.1f}%"
    margin = f"{max(cfg.keep_margin_seconds, cfg.min_pause_seconds / 2):.2f}s"

    cmd = [
        exe, str(audio_path),
        "--edit", f"audio:threshold={threshold_pct}",
        "--margin", margin,
        "--output", str(out_path),
        "--no-open", "--quiet",
    ]

    proc = run(cmd, desc="auto-editor", check=False)
    if proc.returncode != 0 or not out_path.exists():
        # Fall back to auto-editor's default loudness gate if the threshold
        # expression syntax differs in the installed version.
        log.warning("auto-editor invocation failed (%s); retrying with defaults.",
                    proc.returncode)
        cmd_min = [
            exe, str(audio_path),
            "--margin", margin,
            "--output", str(out_path),
            "--no-open",
        ]
        run(cmd_min, desc="auto-editor (defaults)")

    if not out_path.exists():
        raise PipelineError(
            "auto-editor did not produce an output file. Check the auto-editor "
            "version and the silence threshold setting."
        )

    if progress:
        progress(1.0, "Pauses trimmed")
    log.info("Stage 4 complete -> %s", out_path)
    return out_path
