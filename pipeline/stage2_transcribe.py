"""Stage 2 — Transcription via faster-whisper.

Produces word-level timestamps as structured JSON. This word list is the
backbone every later stage consumes (stutter detection, alignment, captions).
"""
from __future__ import annotations

import logging
from pathlib import Path

from .config import TranscribeConfig
from .gpu import device_string, free_gpu_memory, is_oom_error
from .utils import PipelineError, write_json

log = logging.getLogger(__name__)


def transcribe(audio_path: Path, out_json: Path, cfg: TranscribeConfig,
               progress=None) -> dict:
    """Transcribe ``audio_path`` and write a word-timestamp JSON to ``out_json``.

    Returns the parsed dict::

        {
          "language": "en",
          "duration": 123.4,
          "words": [{"idx": 0, "word": "Hello", "start": 0.1, "end": 0.4, "prob": 0.98}, ...],
          "text": "full transcript ..."
        }
    """
    audio_path = Path(audio_path)
    out_json = Path(out_json)

    try:
        from faster_whisper import WhisperModel
    except Exception as exc:  # pragma: no cover
        raise PipelineError(
            "faster-whisper is not installed. `pip install -r requirements.txt`. "
            f"Underlying error: {exc}"
        ) from exc

    device = device_string(prefer_cuda=(cfg.device == "cuda"))
    compute_type = cfg.compute_type if device == "cuda" else "int8"

    if progress:
        progress(0.05, f"Loading {cfg.model_size} ({compute_type})")

    try:
        model = WhisperModel(cfg.model_size, device=device, compute_type=compute_type)
    except Exception as exc:
        if is_oom_error(exc):
            raise PipelineError(
                f"GPU OOM loading Whisper '{cfg.model_size}'. On 6 GB VRAM try a "
                f"smaller model ('distil-large-v3' or 'medium') or compute_type "
                f"'int8' in Settings."
            ) from exc
        raise PipelineError(f"Failed to load Whisper model: {exc}") from exc

    if progress:
        progress(0.2, "Transcribing")

    try:
        segments, info = model.transcribe(
            str(audio_path),
            word_timestamps=True,
            beam_size=cfg.beam_size,
            language=cfg.language,
            vad_filter=cfg.vad_filter,
        )

        words: list[dict] = []
        text_parts: list[str] = []
        for seg in segments:  # generator — consuming it runs the decode
            if seg.words is None:
                continue
            for w in seg.words:
                words.append({
                    "idx": len(words),
                    "word": w.word.strip(),
                    "start": round(float(w.start), 3),
                    "end": round(float(w.end), 3),
                    "prob": round(float(w.probability), 4),
                })
            text_parts.append(seg.text.strip())
            if progress and info.duration:
                progress(min(0.9, 0.2 + 0.7 * (seg.end / info.duration)), "Transcribing")
    except RuntimeError as exc:
        if is_oom_error(exc):
            raise PipelineError(
                "GPU OOM during transcription. Switch to a smaller Whisper model "
                "or compute_type 'int8' in Settings."
            ) from exc
        raise
    finally:
        del model
        free_gpu_memory()

    if not words:
        raise PipelineError(
            "Transcription produced no words. Check the audio is speech and not "
            "silent, or disable the VAD filter in Settings."
        )

    result = {
        "language": info.language,
        "duration": round(float(info.duration), 3),
        "words": words,
        "text": " ".join(text_parts).strip(),
    }
    write_json(out_json, result)
    if progress:
        progress(1.0, f"Transcribed {len(words)} words")
    log.info("Stage 2 complete: %d words -> %s", len(words), out_json)
    return result
