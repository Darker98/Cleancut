"""Central configuration for the pipeline.

Every tunable lives here as a dataclass so the Gradio frontend, the CLI, and the
stage modules all share one source of truth. Paths use :class:`pathlib.Path`
throughout so the code is Windows-safe.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Project-relative locations
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
FONTS_DIR = ASSETS_DIR / "fonts"
DEFAULT_WORK_DIR = PROJECT_ROOT / "work"

# Bundled caption font. The .ass style references it by *name*; libass resolves
# the name through fontconfig, but we also pass the font directory explicitly to
# ffmpeg via ``fontsdir=`` so a system install is never required.
DEFAULT_FONT_FILE = FONTS_DIR / "Montserrat-Bold.ttf"
DEFAULT_FONT_NAME = "Montserrat"  # family name as embedded in the .ttf


@dataclass
class EnhanceConfig:
    """Stage 1 — resemble-enhance."""

    # NOTE: resemble-enhance 0.0.1's enhance()/denoise() chunk audio internally
    # (~30s windows) and do NOT accept these as kwargs, so they are not passed
    # through. Kept for reference / future versions that expose chunking.
    chunk_seconds: float = 10.0
    chunks_overlap: float = 1.0
    nfe: int = 32                      # diffusion steps; lower = faster, less VRAM
    solver: str = "midpoint"           # midpoint | rk4 | euler
    lambd: float = 0.9
    tau: float = 0.5
    denoise_only: bool = False         # True = denoise without full enhancement
    target_sample_rate: int = 44100    # output sample rate (resemble runs at 44.1k)


@dataclass
class TranscribeConfig:
    """Stage 2 — faster-whisper."""

    # Default large-v3; documented fallbacks for 6 GB VRAM are distil-large-v3
    # and medium (see README). The frontend exposes these as a dropdown.
    model_size: str = "large-v3"
    # int8_float16 keeps large-v3 under the 6 GB ceiling. Fall back to "int8"
    # if even that OOMs, or "float16" only on >=10 GB cards.
    compute_type: str = "int8_float16"
    device: str = "cuda"
    language: str | None = None        # None = auto-detect
    beam_size: int = 5
    vad_filter: bool = True            # drop non-speech before decoding


@dataclass
class StutterConfig:
    """Stage 3 — stutter / false-start detection."""

    # Heuristic n-gram repetition window.
    ngram_min: int = 2
    ngram_max: int = 6
    fuzzy_threshold: int = 85          # rapidfuzz ratio (0-100) to call a repeat
    window_seconds: float = 6.0        # only compare phrases within this gap
    restart_phrases: list[str] = field(
        default_factory=lambda: [
            "let me start over",
            "let me restart",
            "let me try that again",
            "let me redo that",
            "wait actually",
            "wait no",
            "sorry i mean",
            "sorry let me",
            "i mean",
            "scratch that",
            "take two",
            "one more time",
            "hold on",
        ]
    )
    # Groq verification.
    groq_model: str = "llama-3.3-70b-versatile"
    groq_context_words: int = 25       # words of context on each side sent to Groq
    use_groq: bool = True              # if False, surface heuristic candidates only


@dataclass
class PauseConfig:
    """Stage 4 — auto-editor silence trimming."""

    silence_threshold: float = 0.04    # loudness threshold (auto-editor "4%")
    min_pause_seconds: float = 0.7     # only cut silences longer than this
    keep_margin_seconds: float = 0.15  # padding kept on each side of speech (~0.3s total)


@dataclass
class BackgroundConfig:
    """Stage 5 — background clip assembly."""

    segment_seconds: float = 60.0      # max length of one background segment
    order: str = "sequential"          # sequential | random
    random_seed: int = 1234            # deterministic "random" ordering
    crossfade_seconds: float = 2.0     # soft dissolve when switching clips (0 = hard cut)


@dataclass
class CaptionConfig:
    """Stage 6 — caption styling (.ass)."""

    max_words: int = 7                 # upper bound per caption event
    min_words: int = 3                 # preferred lower bound
    max_duration: float = 2.4          # seconds before switching chunk
    font_name: str = DEFAULT_FONT_NAME
    font_file: Path = DEFAULT_FONT_FILE
    # Sizing/position are expressed as fractions of the *visible* frame height
    # (i.e. excluding the letterbox bars) and resolved to ASS pixels at render.
    font_height_frac: float = 0.065    # ~6.5% of visible frame height per line
    bottom_margin_frac: float = 0.09   # baseline ~9% up from visible-frame bottom
    outline_px_frac: float = 0.0035    # outline thickness as frac of visible height
    primary_colour: str = "&H00FFFFFF"  # pure white (ASS &HAABBGGRR)
    outline_colour: str = "&H00000000"  # solid black outline (used when box off)
    shadow_px: int = 0                 # 0 = outline only (no drop shadow)
    # Semi-opaque background box behind the caption text (improves legibility).
    box_enabled: bool = True
    box_opacity: float = 0.5           # 0 = transparent, 1 = fully opaque
    box_padding_frac: float = 0.012    # box padding as frac of visible height
    # Rough character width relative to font height — used to decide when a
    # single-line caption would overflow and must be split into two events.
    avg_char_width_frac: float = 0.52


@dataclass
class LetterboxConfig:
    """Stage 8 — cinematic letterbox."""

    aspect_ratio: float = 2.4 / 1.0    # target width:height
    pad_colour: str = "black"


@dataclass
class RenderConfig:
    """Stage 7 — final mux / encode."""

    width: int = 1920                  # canvas width before letterbox padding
    height: int = 1080                 # source picture height (pre-pad)
    fps: int = 30
    video_codec: str = "libx264"       # swap to h264_nvenc for GPU encode
    crf: int = 18
    preset: str = "medium"
    audio_codec: str = "aac"
    audio_bitrate: str = "192k"


@dataclass
class PipelineConfig:
    """Top-level config aggregating every stage + run-wide settings."""

    work_dir: Path = DEFAULT_WORK_DIR
    keep_intermediate: bool = True

    enhance: EnhanceConfig = field(default_factory=EnhanceConfig)
    transcribe: TranscribeConfig = field(default_factory=TranscribeConfig)
    stutter: StutterConfig = field(default_factory=StutterConfig)
    pauses: PauseConfig = field(default_factory=PauseConfig)
    background: BackgroundConfig = field(default_factory=BackgroundConfig)
    captions: CaptionConfig = field(default_factory=CaptionConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    letterbox: LetterboxConfig = field(default_factory=LetterboxConfig)

    def run_dir(self, run_id: str) -> Path:
        """Per-run scratch directory holding every stage's intermediate output."""
        d = self.work_dir / run_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def groq_api_key(self) -> str | None:
        return os.environ.get("GROQ_API_KEY")
