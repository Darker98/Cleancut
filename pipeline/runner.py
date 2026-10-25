"""Pipeline orchestration.

Two phases, matching the Gradio flow:

* :meth:`PipelineRunner.run_analysis` — Stage 1 (enhance) → Stage 2 (transcribe)
  → Stage 3 (flag stutter candidates). Stops for user review.
* :meth:`PipelineRunner.run_assembly` — apply the *accepted* cuts → Stage 4
  (pause trim) → re-transcribe the cleaned audio for authoritative timestamps →
  Stage 5 (background) → Stage 6 (captions) → Stage 7/8 (composite + letterbox).

GPU-heavy stages (1, 2, and the re-transcribe) run sequentially and release VRAM
between them so they never have to coexist in 6 GB.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from . import (
    align,
    stage1_prepare,
    stage2_transcribe,
    stage3_stutter,
    stage4_pauses,
    stage5_background,
    stage6_captions,
    stage7_composite,
)
from .config import PipelineConfig
from .gpu import free_gpu_memory
from .stage8_letterbox import compute_geometry
from .utils import (
    PipelineError,
    ensure_ffmpeg_has_libass,
    list_video_files,
    probe_duration,
    read_json,
    write_json,
)

log = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    run_id: str
    run_dir: Path
    audio: Path
    transcript: dict
    candidates: list[dict]

    def to_state(self) -> dict:
        return {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "audio": str(self.audio),
            "transcript_path": str(self.run_dir / "transcript.json"),
            "candidates_path": str(self.run_dir / "stutter_candidates.json"),
        }


def _sub_progress(progress, lo: float, hi: float):
    """Map a stage's 0..1 progress into the [lo, hi] slice of the whole run."""
    if progress is None:
        return None

    def inner(frac: float, desc: str = ""):
        progress(lo + (hi - lo) * max(0.0, min(1.0, frac)), desc)

    return inner


class PipelineRunner:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

    # ------------------------------------------------------------------ #
    # Phase A — analysis (Stages 1–3)
    # ------------------------------------------------------------------ #
    def run_analysis(self, input_audio: Path, run_id: str, progress=None) -> AnalysisResult:
        ensure_ffmpeg_has_libass()
        cfg = self.cfg
        run_dir = cfg.run_dir(run_id)
        input_audio = Path(input_audio)
        if not input_audio.exists():
            raise PipelineError(f"Input audio not found: {input_audio}")

        # Stage 1 — prepare original audio (no enhancement; keeps the raw voice)
        prepared = run_dir / "01_audio.wav"
        stage1_prepare.prepare_audio(
            input_audio, prepared,
            progress=_sub_progress(progress, 0.0, 0.1),
        )

        # Stage 2 — transcribe (GPU)
        transcript = stage2_transcribe.transcribe(
            prepared, run_dir / "transcript.json", cfg.transcribe,
            progress=_sub_progress(progress, 0.1, 0.9),
        )
        free_gpu_memory()

        # Stage 3 — flag stutter candidates (CPU + Groq)
        candidates = stage3_stutter.detect_stutters(
            transcript, run_dir / "stutter_candidates.json", cfg.stutter,
            cfg.groq_api_key,
            progress=_sub_progress(progress, 0.9, 1.0),
        )

        result = AnalysisResult(run_id, run_dir, prepared, transcript, candidates)
        # 'prepared' is the unprocessed original voice; downstream stages cut and
        # mux this exact audio so the final video keeps the raw recording.
        write_json(run_dir / "state.json", result.to_state())
        return result

    # ------------------------------------------------------------------ #
    # Phase B — assembly (apply cuts, Stages 4–8)
    # ------------------------------------------------------------------ #
    def run_assembly(self, state: dict, candidates: list[dict], script: str,
                     clips_folder: Path, run_id: str, progress=None) -> Path:
        cfg = self.cfg
        run_dir = Path(state["run_dir"])
        source_audio = Path(state["audio"])
        transcript = read_json(state["transcript_path"])
        clips = list_video_files(Path(clips_folder))

        # Stage 3 (apply) — render accepted cuts on the original audio.
        if progress:
            progress(0.05, "Applying accepted cuts")
        cut_audio = run_dir / "03_cut.wav"
        stage3_stutter.apply_accepted_cuts(source_audio, transcript, candidates, cut_audio)

        # Stage 4 — pause trim.
        cleaned = run_dir / "04_cleaned.wav"
        stage4_pauses.trim_pauses(
            cut_audio, cleaned, cfg.pauses,
            progress=_sub_progress(progress, 0.05, 0.2),
        )

        # Re-transcribe the cleaned audio for authoritative final timestamps
        # (the timeline changed after cuts + pause trimming).
        final_transcript = stage2_transcribe.transcribe(
            cleaned, run_dir / "transcript_final.json", cfg.transcribe,
            progress=_sub_progress(progress, 0.2, 0.4),
        )
        free_gpu_memory()

        total_duration = probe_duration(cleaned)

        # Align script [BREAK] markers → segment list.
        break_times = align.align_breaks(script, final_transcript)
        segments = align.build_segments(
            break_times, total_duration, cfg.background.segment_seconds
        )
        write_json(run_dir / "segments.json",
                   {"break_times": break_times, "segments": segments,
                    "duration": total_duration})

        # Stage 5 — background assembly.
        background = run_dir / "05_background.mp4"
        stage5_background.assemble_background(
            segments, clips, background, cfg.background, cfg.render,
            work=run_dir / "bg_work",
            progress=_sub_progress(progress, 0.4, 0.75),
        )

        # Stage 6 — captions.
        geo = compute_geometry(cfg.render, cfg.letterbox)
        ass_file = run_dir / "06_captions.ass"
        stage6_captions.build_ass(final_transcript["words"], ass_file, cfg.captions, geo)
        events = read_json(ass_file.with_suffix(".events.json"))
        if progress:
            progress(0.8, "Captions generated")

        # Stage 7/8 — composite + letterbox.
        final = run_dir / "final.mp4"
        stage7_composite.compose_final(
            background, cleaned, ass_file, final, geo,
            cfg.render, cfg.letterbox, cfg.captions.font_file.parent,
            progress=_sub_progress(progress, 0.8, 1.0),
        )
        write_json(run_dir / "final_state.json", {"final": str(final)})
        log.info("Pipeline complete -> %s", final)
        # Return the final video plus the caption events so the frontend can
        # offer editing + a fast re-render (Stage 6/7 only) without redoing the
        # transcription or background assembly.
        return final, events

    def rerender_captions(self, run_id: str, events: list[dict],
                          progress=None) -> Path:
        """Rebuild captions from (edited) events and re-composite.

        Reuses the cached background track and cut audio from the original
        assembly run, so this is a single ffmpeg pass — no re-transcription or
        background rebuild.
        """
        cfg = self.cfg
        run_dir = cfg.run_dir(run_id)
        background = run_dir / "05_background.mp4"
        cleaned = run_dir / "04_cleaned.wav"
        if not background.exists() or not cleaned.exists():
            raise PipelineError(
                "Cannot re-render captions: the assembled background/audio for "
                "this run is missing. Run the full assembly step first."
            )

        if progress:
            progress(0.2, "Rebuilding captions")
        geo = compute_geometry(cfg.render, cfg.letterbox)
        ass_file = run_dir / "06_captions.ass"
        stage6_captions.build_ass_from_events(events, ass_file, cfg.captions, geo)

        final = run_dir / "final.mp4"
        stage7_composite.compose_final(
            background, cleaned, ass_file, final, geo,
            cfg.render, cfg.letterbox, cfg.captions.font_file.parent,
            progress=_sub_progress(progress, 0.3, 1.0),
        )
        log.info("Re-rendered captions -> %s", final)
        return final
