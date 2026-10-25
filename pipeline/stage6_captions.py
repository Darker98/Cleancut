"""Stage 6 — Caption generation (.ass).

Phrase-level chunks (≈3–7 words), each a single line shown for ≤ ~2.4s, in
sentence case, white text with a solid black outline, no background box. Size
and vertical position are computed against the *visible* picture height (from
:mod:`pipeline.stage8_letterbox`) so captions never sit inside the black bars.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from .config import CaptionConfig
from .stage8_letterbox import LetterboxGeometry
from .utils import format_timestamp, write_json

log = logging.getLogger(__name__)

_SENTENCE_END = tuple(".!?")
_PHRASE_END = tuple(",;:") + _SENTENCE_END


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
def _max_chars_per_line(geo: LetterboxGeometry, font_px: float,
                        cfg: CaptionConfig) -> int:
    """How many characters fit on one centered line at this font size."""
    usable_w = geo.visible_w * 0.9          # 5% safe margin each side
    char_w = cfg.avg_char_width_frac * font_px
    return max(8, int(usable_w / char_w))


def _split_to_width(words: list[dict], max_chars: int) -> list[list[dict]]:
    """Split a word list into runs that each fit on a single line."""
    runs: list[list[dict]] = []
    current: list[dict] = []
    length = 0
    for w in words:
        token = w["word"]
        add = len(token) + (1 if current else 0)
        if current and length + add > max_chars:
            runs.append(current)
            current, length = [], 0
            add = len(token)
        current.append(w)
        length += add
    if current:
        runs.append(current)
    return runs


def build_chunks(words: list[dict], cfg: CaptionConfig,
                 geo: LetterboxGeometry, font_px: float) -> list[dict]:
    """Group words into caption events honouring word/duration/line limits."""
    max_chars = _max_chars_per_line(geo, font_px, cfg)
    raw_chunks: list[list[dict]] = []
    current: list[dict] = []

    def flush():
        if current:
            raw_chunks.append(current.copy())
            current.clear()

    for w in words:
        if not w["word"].strip():
            continue
        if current:
            span = w["end"] - current[0]["start"]
            ends_phrase = current[-1]["word"].endswith(_PHRASE_END)
            if (len(current) >= cfg.max_words
                    or span > cfg.max_duration
                    or (ends_phrase and len(current) >= cfg.min_words)):
                flush()
        current.append(w)
    flush()

    # Enforce the single-line rule: split any chunk too wide for one line into
    # separate events, dividing the chunk's timespan proportionally by words.
    events: list[dict] = []
    for chunk in raw_chunks:
        runs = _split_to_width(chunk, max_chars)
        for run in runs:
            events.append({
                "start": run[0]["start"],
                "end": run[-1]["end"],
                "text": " ".join(r["word"] for r in run).strip(),
            })

    # Clamp each event to max_duration and prevent overlaps.
    for i, ev in enumerate(events):
        ev["end"] = min(ev["end"], ev["start"] + cfg.max_duration)
        if i + 1 < len(events):
            ev["end"] = min(ev["end"], events[i + 1]["start"])
        if ev["end"] <= ev["start"]:
            ev["end"] = ev["start"] + 0.4
    return events


# --------------------------------------------------------------------------- #
# ASS rendering
# --------------------------------------------------------------------------- #
def _ass_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "(").replace("}", ")")


def _to_sentence_case(text: str) -> str:
    """Light sentence-case normalisation (Whisper already mixed-cases well)."""
    return text  # Whisper output is already sentence case; preserve as-is.


def _alpha_hex(opacity: float) -> str:
    """ASS alpha byte for a given *opacity* (0..1). ASS: 00=opaque, FF=clear."""
    opacity = max(0.0, min(1.0, opacity))
    return f"{round((1.0 - opacity) * 255):02X}"


def _style_line(cfg: CaptionConfig, geo: LetterboxGeometry) -> str:
    """Build the [V4+ Styles] Default line, with optional semi-opaque box."""
    font_px = round(cfg.font_height_frac * geo.visible_h)
    margin_v = round(geo.bar_h + cfg.bottom_margin_frac * geo.visible_h)

    if cfg.box_enabled:
        # BorderStyle 3 = opaque box drawn in OutlineColour; "Outline" is the box
        # padding. Use a semi-transparent black box for legibility w/o a hard bar.
        box_colour = f"&H{_alpha_hex(cfg.box_opacity)}000000"  # &HAABBGGRR, black
        border_style = 3
        outline_val = max(1, round(cfg.box_padding_frac * geo.visible_h))
        outline_colour = box_colour
        back_colour = box_colour
        shadow_val = 0
    else:
        border_style = 1                                       # outline + shadow
        outline_val = max(1, round(cfg.outline_px_frac * geo.visible_h))
        outline_colour = cfg.outline_colour
        back_colour = "&H00000000"
        shadow_val = cfg.shadow_px

    # &HAABBGGRR colours. Alignment 2 = bottom-center. Bold = -1.
    return (
        f"Style: Default,{cfg.font_name},{font_px},"
        f"{cfg.primary_colour},&H000000FF,{outline_colour},{back_colour},"
        f"-1,0,0,0,100,100,0,0,{border_style},{outline_val},{shadow_val},"
        f"2,40,40,{margin_v},1"
    )


def build_ass(words: list[dict], out_path: Path, cfg: CaptionConfig,
              geo: LetterboxGeometry) -> Path:
    """Compute caption events from words and write a styled .ass file."""
    font_px = round(cfg.font_height_frac * geo.visible_h)
    events = build_chunks(words, cfg, geo, font_px)
    return build_ass_from_events(events, out_path, cfg, geo)


def build_ass_from_events(events: list[dict], out_path: Path, cfg: CaptionConfig,
                          geo: LetterboxGeometry) -> Path:
    """Write a styled .ass file from an explicit event list.

    Used both for the initial auto-generated captions and to re-render after the
    user edits caption text/timings in the frontend.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "WrapStyle: 2\n"           # no automatic wrapping — we control line breaks
        "ScaledBorderAndShadow: yes\n"
        f"PlayResX: {geo.canvas_w}\n"
        f"PlayResY: {geo.canvas_h}\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"{_style_line(cfg, geo)}\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, "
        "MarginV, Effect, Text\n"
    )

    lines = [header]
    clean_events = []
    for ev in events:
        text = str(ev.get("text", "")).strip()
        if not text:
            continue
        start = float(ev["start"])
        end = float(ev["end"])
        if end <= start:
            end = start + 0.4
        clean_events.append({"start": start, "end": end, "text": text})
        lines.append(
            f"Dialogue: 0,{format_timestamp(start)},{format_timestamp(end)},"
            f"Default,,0,0,0,,{_ass_escape(_to_sentence_case(text))}\n"
        )

    out_path.write_text("".join(lines), encoding="utf-8")
    # Persist the event list for the frontend's caption editor.
    write_json(out_path.with_suffix(".events.json"), clean_events)
    log.info("Stage 6: wrote %d caption events -> %s", len(clean_events), out_path)
    return out_path
