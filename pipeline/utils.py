"""Shared utilities: ffmpeg/ffprobe wrappers, JSON IO, logging helpers."""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, Sequence

log = logging.getLogger(__name__)


class PipelineError(RuntimeError):
    """Raised for any recoverable, user-facing pipeline failure."""


# --------------------------------------------------------------------------- #
# Binary discovery
# --------------------------------------------------------------------------- #
def find_binary(name: str) -> str:
    """Locate an executable on PATH, raising a clear error if missing."""
    exe = shutil.which(name)
    if not exe:
        raise PipelineError(
            f"`{name}` was not found on your PATH. "
            f"Install it and reopen the terminal. See the README setup section."
        )
    return exe


def ensure_ffmpeg_has_libass() -> None:
    """Confirm the installed ffmpeg was built with libass (needed to burn .ass)."""
    ffmpeg = find_binary("ffmpeg")
    out = subprocess.run(
        [ffmpeg, "-hide_banner", "-version"],
        capture_output=True, text=True,
    )
    if "libass" not in (out.stdout + out.stderr):
        raise PipelineError(
            "Your ffmpeg build does not include libass, which is required to "
            "burn .ass captions. Install a full build (e.g. the gyan.dev "
            "'full' build) and update PATH. See README."
        )


# --------------------------------------------------------------------------- #
# Subprocess runner
# --------------------------------------------------------------------------- #
def run(cmd: Sequence[str], *, desc: str = "", check: bool = True) -> subprocess.CompletedProcess:
    """Run a subprocess, capturing output and surfacing failures clearly."""
    cmd = [str(c) for c in cmd]
    log.debug("RUN %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if check and proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-25:]
        raise PipelineError(
            f"{desc or cmd[0]} failed (exit {proc.returncode}).\n" + "\n".join(tail)
        )
    return proc


def run_ffmpeg(args: Sequence[str], *, desc: str = "ffmpeg") -> None:
    """Run ffmpeg with -y and quiet logging; args exclude the binary name."""
    ffmpeg = find_binary("ffmpeg")
    run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error", *args], desc=desc)


# --------------------------------------------------------------------------- #
# ffprobe helpers
# --------------------------------------------------------------------------- #
def probe_duration(path: Path) -> float:
    """Return media duration in seconds."""
    ffprobe = find_binary("ffprobe")
    proc = run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        desc="ffprobe duration",
    )
    try:
        return float(proc.stdout.strip())
    except ValueError as exc:
        raise PipelineError(f"Could not read duration of {path}: {proc.stdout!r}") from exc


def probe_resolution(path: Path) -> tuple[int, int]:
    """Return (width, height) of the first video stream."""
    ffprobe = find_binary("ffprobe")
    proc = run(
        [ffprobe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height",
         "-of", "csv=s=x:p=0", str(path)],
        desc="ffprobe resolution",
    )
    try:
        w, h = proc.stdout.strip().split("x")
        return int(w), int(h)
    except ValueError as exc:
        raise PipelineError(f"Could not read resolution of {path}: {proc.stdout!r}") from exc


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".wmv", ".flv"}


def list_video_files(folder: Path) -> list[Path]:
    """Return sorted video files in a folder (non-recursive)."""
    if not folder.exists() or not folder.is_dir():
        raise PipelineError(f"Background clips folder not found: {folder}")
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in VIDEO_EXTS)
    if not files:
        raise PipelineError(
            f"No video clips found in {folder}. Supported: {', '.join(sorted(VIDEO_EXTS))}"
        )
    return files


# --------------------------------------------------------------------------- #
# JSON IO
# --------------------------------------------------------------------------- #
def write_json(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def format_timestamp(seconds: float) -> str:
    """Format seconds as H:MM:SS.cc for ASS timing."""
    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:d}:{m:02d}:{s:05.2f}"
