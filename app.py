"""Cleancut: Gradio frontend for the local YouTube voiceover post-production pipeline.

Single local web app (no auth) that walks through: upload → settings → run
analysis (Stages 1–3) → review flagged stutters → continue (Stages 4–8) →
preview & download.
"""
from __future__ import annotations

import logging
import traceback
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv


def _patch_gradio_client_schema_bug() -> None:
    """Work around a gradio 4.44 / gradio_client 1.3 bug.

    ``gradio_client.utils._json_schema_to_python_type`` recurses into schema
    values without checking for booleans, so a schema like
    ``{"additionalProperties": true}`` (which our dict-holding gr.State
    components generate) hits ``if "const" in schema`` with ``schema=True`` and
    raises ``TypeError: argument of type 'bool' is not iterable``. That crashes
    the API-info endpoint, which makes the localhost health check 500 and aborts
    launch. Teaching the function to treat a bool schema as a plain bool type
    fixes it. Safe no-op on versions where the bug is already fixed.
    """
    try:
        import gradio_client.utils as gcu
    except Exception:  # pragma: no cover
        return

    _orig = gcu._json_schema_to_python_type

    def _safe(schema, defs=None):
        if isinstance(schema, bool):
            return "bool"
        return _orig(schema, defs)

    gcu._json_schema_to_python_type = _safe

    if hasattr(gcu, "get_type"):
        _orig_get_type = gcu.get_type

        def _safe_get_type(schema):
            if isinstance(schema, bool):
                return "bool"
            return _orig_get_type(schema)

        gcu.get_type = _safe_get_type


_patch_gradio_client_schema_bug()

from pipeline.config import (
    DEFAULT_FONT_FILE,
    PipelineConfig,
)
from pipeline.gpu import describe_environment
from pipeline.runner import PipelineRunner
from pipeline.utils import PipelineError, run_ffmpeg

load_dotenv()
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

WHISPER_MODELS = ["large-v3", "distil-large-v3", "medium", "small", "base"]
COMPUTE_TYPES = ["int8_float16", "int8", "float16"]


# --------------------------------------------------------------------------- #
# Config assembly from UI
# --------------------------------------------------------------------------- #
def build_config(model_size, compute_type, pause_thresh, min_pause,
                 max_words, max_dur, segment_len, aspect_w, aspect_h,
                 clip_order, use_groq, crossfade, box_opacity) -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.transcribe.model_size = model_size
    cfg.transcribe.compute_type = compute_type
    cfg.pauses.silence_threshold = float(pause_thresh)
    cfg.pauses.min_pause_seconds = float(min_pause)
    cfg.captions.max_words = int(max_words)
    cfg.captions.max_duration = float(max_dur)
    cfg.background.segment_seconds = float(segment_len)
    cfg.background.order = clip_order
    cfg.background.crossfade_seconds = float(crossfade)
    cfg.letterbox.aspect_ratio = float(aspect_w) / float(aspect_h)
    cfg.stutter.use_groq = bool(use_groq)
    cfg.captions.box_opacity = float(box_opacity)
    cfg.captions.box_enabled = float(box_opacity) > 0.0
    return cfg


def _events_to_rows(events: list[dict]) -> list[list]:
    return [[round(float(e["start"]), 2), round(float(e["end"]), 2), e["text"]]
            for e in events]


def _rows_to_events(rows) -> list[dict]:
    """Coerce the caption Dataframe (list of [start, end, text] rows) to events."""
    events = []
    for row in (rows or []):
        if row is None or len(row) < 3:
            continue
        text = str(row[2]).strip()
        if not text:
            continue
        try:
            start = float(row[0])
            end = float(row[1])
        except (TypeError, ValueError):
            continue
        events.append({"start": start, "end": end, "text": text})
    events.sort(key=lambda e: e["start"])
    return events


def _extract_snippet(audio_path: Path, start: float, end: float,
                     dst: Path, pad: float = 0.8) -> Path:
    s = max(0.0, start - pad)
    dur = (end - start) + 2 * pad
    run_ffmpeg(
        ["-ss", f"{s:.3f}", "-i", str(audio_path), "-t", f"{dur:.3f}", str(dst)],
        desc="extract review snippet",
    )
    return dst


# --------------------------------------------------------------------------- #
# Phase A — analysis
# --------------------------------------------------------------------------- #
def _status(msg: str):
    return gr.update(value=msg, visible=True)


def run_analysis(audio_file, model_size, compute_type, pause_thresh, min_pause,
                 max_words, max_dur, segment_len, aspect_w, aspect_h,
                 clip_order, use_groq, crossfade, box_opacity,
                 progress=gr.Progress()):
    # outputs: [state, candidates_state, status, review_group, continue_btn]
    if not audio_file:
        raise gr.Error("Please upload a voiceover recording first.")

    # Immediate feedback the instant the button is pressed.
    yield (gr.update(), gr.update(),
           _status("⏳ **Starting…** Stage 1/3 preparing audio → 2/3 transcribing "
                   "(slowest step) → 3/3 scanning for stutters. Live status shows "
                   "in the progress bar."),
           gr.update(visible=False), gr.update(visible=False))

    try:
        cfg = build_config(model_size, compute_type, pause_thresh, min_pause,
                           max_words, max_dur, segment_len, aspect_w, aspect_h,
                           clip_order, use_groq, crossfade, box_opacity)
        runner = PipelineRunner(cfg)
        run_id = Path(audio_file).stem.replace(" ", "_") + "_run"

        def prog(frac, desc=""):
            progress(frac, desc=desc)

        result = runner.run_analysis(Path(audio_file), run_id, progress=prog)

        progress(0.98, desc="Building review clips")
        for c in result.candidates:
            snip = result.run_dir / f"snippet_{c['id']}.wav"
            _extract_snippet(result.audio, c["start_time"], c["end_time"], snip)
            c["snippet"] = str(snip)

        state = result.to_state()
        n = len(result.candidates)
        msg = (f"✅ **Analysis complete** — {n} stutter candidate(s) flagged. "
               f"Review them below, then click **Continue**." if n else
               "✅ **Analysis complete** — no stutters flagged. Click **Continue**.")
        yield (state, result.candidates, _status(msg),
               gr.update(visible=True), gr.update(visible=True))
    except PipelineError as e:
        raise gr.Error(str(e))
    except Exception as e:  # noqa: BLE001
        log.error("Analysis failed:\n%s", traceback.format_exc())
        raise gr.Error(f"Unexpected error during analysis: {e}")


# --------------------------------------------------------------------------- #
# Phase B — assembly
# --------------------------------------------------------------------------- #
def run_assembly(state, candidates, script, clips_folder, model_size, compute_type,
                 pause_thresh, min_pause, max_words, max_dur, segment_len,
                 aspect_w, aspect_h, clip_order, use_groq, crossfade, box_opacity,
                 progress=gr.Progress()):
    # outputs: [out_status, video_out, file_out, captions_df, captions_group,
    #           rerender_btn]
    def _out(status, video=gr.update(), file=gr.update(), df=gr.update(),
             group=gr.update(), rerender=gr.update()):
        return (status, video, file, df, group, rerender)

    if not state:
        raise gr.Error("Run the analysis step first.")
    if not script or not script.strip():
        raise gr.Error("Please paste your script (with [BREAK] markers).")
    if not clips_folder or not Path(clips_folder).exists():
        raise gr.Error("Please provide a valid background clips folder path.")

    yield _out(_status("⏳ **Rendering…** applying cuts → trimming pauses → "
                       "re-transcribing → assembling background (with crossfades) "
                       "→ captions → final composite. Watch the progress bar."),
               group=gr.update(visible=False), rerender=gr.update(visible=False))

    try:
        cfg = build_config(model_size, compute_type, pause_thresh, min_pause,
                           max_words, max_dur, segment_len, aspect_w, aspect_h,
                           clip_order, use_groq, crossfade, box_opacity)
        runner = PipelineRunner(cfg)

        def prog(frac, desc=""):
            progress(frac, desc=desc)

        final, events = runner.run_assembly(
            state, candidates or [], script, Path(clips_folder),
            state["run_id"], progress=prog,
        )
        yield _out(
            _status("✅ **Render complete.** Spot a caption mistake? Edit the text "
                    "(or timings) in the table below and click **Re-render "
                    "captions** — it only re-does the caption pass, so it's quick."),
            video=gr.update(value=str(final), visible=True),
            file=gr.update(value=str(final), visible=True),
            df=gr.update(value=_events_to_rows(events)),
            group=gr.update(visible=True),
            rerender=gr.update(visible=True),
        )
    except PipelineError as e:
        raise gr.Error(str(e))
    except Exception as e:  # noqa: BLE001
        log.error("Assembly failed:\n%s", traceback.format_exc())
        raise gr.Error(f"Unexpected error during assembly: {e}")


# --------------------------------------------------------------------------- #
# Caption re-render (after edits)
# --------------------------------------------------------------------------- #
def rerender_captions(state, caption_rows, model_size, compute_type, pause_thresh,
                      min_pause, max_words, max_dur, segment_len, aspect_w,
                      aspect_h, clip_order, use_groq, crossfade, box_opacity,
                      progress=gr.Progress()):
    # outputs: [out_status, video_out, file_out]
    if not state:
        raise gr.Error("Run the pipeline first.")
    events = _rows_to_events(caption_rows)
    if not events:
        raise gr.Error("No caption rows to render.")

    yield (_status("⏳ Re-rendering captions (caption pass only)…"),
           gr.update(), gr.update())

    try:
        cfg = build_config(model_size, compute_type, pause_thresh, min_pause,
                           max_words, max_dur, segment_len, aspect_w, aspect_h,
                           clip_order, use_groq, crossfade, box_opacity)
        runner = PipelineRunner(cfg)

        def prog(frac, desc=""):
            progress(frac, desc=desc)

        final = runner.rerender_captions(state["run_id"], events, progress=prog)
        yield (_status("✅ **Captions re-rendered.**"),
               gr.update(value=str(final), visible=True),
               gr.update(value=str(final), visible=True))
    except PipelineError as e:
        raise gr.Error(str(e))
    except Exception as e:  # noqa: BLE001
        log.error("Caption re-render failed:\n%s", traceback.format_exc())
        raise gr.Error(f"Unexpected error during re-render: {e}")


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def build_ui():
    with gr.Blocks(title="Cleancut",
                   theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎬 Cleancut")
        gr.Markdown(f"*GPU environment:* `{describe_environment()}`")

        state = gr.State(None)            # serialised analysis state
        candidates_state = gr.State([])   # list of candidate dicts (mutable)

        # 1 — Upload
        with gr.Group():
            gr.Markdown("## 1 · Inputs")
            audio_in = gr.File(
                label="Voiceover recording (audio: wav/mp3/m4a/flac/ogg · "
                      "or video: mp4/mkv/mov/webm — the audio track is extracted)",
                type="filepath",
                file_types=[".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac",
                            ".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v"],
            )
            script_in = gr.Textbox(
                label="Script", lines=8,
                placeholder="Paste your script here. Mark topic changes with [BREAK].",
                info="Insert [BREAK] where the background video should switch topics.",
            )
            clips_in = gr.Textbox(
                label="Background clips folder (absolute path)",
                placeholder=r"D:\my-clips",
            )

        # 2 — Settings
        with gr.Accordion("2 · Settings", open=False):
            with gr.Row():
                model_size = gr.Dropdown(WHISPER_MODELS, value="large-v3",
                                         label="Whisper model (↓ for less VRAM)")
                compute_type = gr.Dropdown(COMPUTE_TYPES, value="int8_float16",
                                           label="Compute type")
            with gr.Row():
                pause_thresh = gr.Slider(0.01, 0.2, value=0.04, step=0.01,
                                         label="Silence threshold (loudness)")
                min_pause = gr.Slider(0.2, 2.0, value=0.7, step=0.05,
                                      label="Min pause to cut (s)")
            with gr.Row():
                max_words = gr.Slider(3, 10, value=7, step=1,
                                      label="Caption max words")
                max_dur = gr.Slider(1.0, 4.0, value=2.4, step=0.1,
                                    label="Caption max duration (s)")
            with gr.Row():
                segment_len = gr.Slider(15, 120, value=60, step=5,
                                        label="Background segment length (s)")
                clip_order = gr.Radio(["sequential", "random"], value="sequential",
                                      label="Clip order")
            with gr.Row():
                aspect_w = gr.Number(value=2.4, label="Aspect width")
                aspect_h = gr.Number(value=1.0, label="Aspect height")
                use_groq = gr.Checkbox(value=True, label="Verify stutters with Groq")
            with gr.Row():
                crossfade = gr.Slider(0.0, 4.0, value=2.0, step=0.5,
                                      label="Clip crossfade (s)  ·  0 = hard cut")
                box_opacity = gr.Slider(0.0, 1.0, value=0.5, step=0.05,
                                        label="Caption background opacity  ·  0 = none")

        # 3 — Run analysis
        run_btn = gr.Button("▶ Run analysis (Stages 1–3)", variant="primary")
        status = gr.Markdown(visible=False)

        # 4 — Stutter review
        review_group = gr.Group(visible=False)
        with review_group:
            gr.Markdown("## 4 · Review flagged stutters")
            gr.Markdown(
                "Each card plays just that span (the waveform is shown in the "
                "player). **Accept** to cut it, **Reject** to keep it, or adjust "
                "the cut word range. Pre-accepted = Groq confirmed false start."
            )

            @gr.render(inputs=candidates_state)
            def render_candidates(cands):
                if not cands:
                    gr.Markdown("*No candidates flagged.* 🎉")
                    return
                for i, c in enumerate(cands):
                    groq = c.get("groq", {})
                    verdict = groq.get("is_false_start")
                    badge = ("🟥 Groq: false start" if verdict
                             else "🟩 Groq: intentional" if verdict is False
                             else "🟨 Unverified")
                    with gr.Group():
                        gr.Markdown(
                            f"**#{i + 1} · {badge}** — `{c['kind']}`  \n"
                            f"**Text:** “{c['text']}”  \n"
                            f"*{c['heuristic_reason']}*  \n"
                            f"*Groq:* {groq.get('reasoning', '—')}"
                        )
                        if c.get("snippet"):
                            gr.Audio(value=c["snippet"], label="Span audio",
                                     interactive=False)
                        with gr.Row():
                            decision = gr.Radio(
                                ["Accept (cut)", "Reject (keep)"],
                                value="Accept (cut)" if c.get("accepted")
                                else "Reject (keep)",
                                label="Decision", scale=2,
                            )
                            start_idx = gr.Number(
                                value=c["cut_start_word_idx"],
                                label="Cut start word idx", precision=0, scale=1,
                            )
                            end_idx = gr.Number(
                                value=c["cut_end_word_idx"],
                                label="Cut end word idx", precision=0, scale=1,
                            )

                        def _update(dec, s_idx, e_idx, st, idx=i):
                            st[idx]["accepted"] = dec.startswith("Accept")
                            st[idx]["cut_start_word_idx"] = int(s_idx)
                            st[idx]["cut_end_word_idx"] = int(e_idx)
                            return st

                        for comp in (decision, start_idx, end_idx):
                            comp.change(_update,
                                        [decision, start_idx, end_idx, candidates_state],
                                        candidates_state)

        # 5 — Continue
        continue_btn = gr.Button("▶ Continue — apply cuts & render (Stages 4–8)",
                                 variant="primary", visible=False)

        # 6 — Output
        out_status = gr.Markdown(visible=False)
        video_out = gr.Video(label="Final preview", visible=False)
        file_out = gr.File(label="Download final video", visible=False)

        # 7 — Caption editor (shown after the first render)
        captions_group = gr.Group(visible=False)
        with captions_group:
            gr.Markdown("## 7 · Edit captions")
            gr.Markdown(
                "Spotted a typo or want to tweak timing? Edit the **text** "
                "(or start/end seconds) directly in the table, then click "
                "**Re-render captions**. This only redoes the caption + composite "
                "pass — no re-transcription or background rebuild, so it's fast."
            )
            captions_df = gr.Dataframe(
                headers=["start (s)", "end (s)", "text"],
                datatype=["number", "number", "str"],
                col_count=(3, "fixed"),
                type="array",
                interactive=True,
                wrap=True,
                label="Caption events",
            )
            rerender_btn = gr.Button("♻ Re-render captions", variant="secondary",
                                     visible=False)

        # Wiring
        settings = [model_size, compute_type, pause_thresh, min_pause, max_words,
                    max_dur, segment_len, aspect_w, aspect_h, clip_order, use_groq,
                    crossfade, box_opacity]

        run_btn.click(
            run_analysis,
            inputs=[audio_in, *settings],
            outputs=[state, candidates_state, status, review_group, continue_btn],
        )

        continue_btn.click(
            run_assembly,
            inputs=[state, candidates_state, script_in, clips_in, *settings],
            outputs=[out_status, video_out, file_out, captions_df,
                     captions_group, rerender_btn],
        )

        rerender_btn.click(
            rerender_captions,
            inputs=[state, captions_df, *settings],
            outputs=[out_status, video_out, file_out],
        )

    return demo


if __name__ == "__main__":
    if not DEFAULT_FONT_FILE.exists():
        log.warning("Caption font missing at %s — see assets/fonts/README.md",
                    DEFAULT_FONT_FILE)
    build_ui().queue().launch(server_name="127.0.0.1", inbrowser=True)
