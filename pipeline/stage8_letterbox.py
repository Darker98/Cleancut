"""Stage 8 — Cinematic letterbox geometry + filter.

We keep the full render canvas (e.g. 1920×1080) and mask it down to a 2.4:1
*visible picture* by adding equal black bars top and bottom. Captions (Stage 6)
must sit inside that visible picture, so the geometry helper here is the single
source of truth shared by both stages.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from .config import LetterboxConfig, RenderConfig

log = logging.getLogger(__name__)


@dataclass
class LetterboxGeometry:
    canvas_w: int        # full output width
    canvas_h: int        # full output height
    visible_w: int       # visible picture width (== canvas_w)
    visible_h: int       # visible picture height (canvas_w / aspect)
    bar_h: int           # black bar height (top == bottom)


def compute_geometry(render: RenderConfig, letterbox: LetterboxConfig) -> LetterboxGeometry:
    """Compute the visible-picture geometry for a 2.4:1 letterbox on the canvas."""
    canvas_w, canvas_h = render.width, render.height
    visible_h = int(round(canvas_w / letterbox.aspect_ratio))
    # Visible picture can't exceed the canvas; if the canvas is already wider
    # than the target ratio there are no bars.
    if visible_h >= canvas_h:
        visible_h = canvas_h
    visible_h -= visible_h % 2  # keep even for yuv420p
    bar_h = (canvas_h - visible_h) // 2
    return LetterboxGeometry(
        canvas_w=canvas_w, canvas_h=canvas_h,
        visible_w=canvas_w, visible_h=visible_h, bar_h=bar_h,
    )


def letterbox_filter(geo: LetterboxGeometry, pad_colour: str = "black") -> str:
    """ffmpeg -vf string that crops to the visible band then re-pads with bars.

    Cropping the canvas to the centered 2.4:1 band and padding it back keeps the
    content at native scale (cinematic crop) with clean black bars.
    """
    if geo.bar_h <= 0:
        return "null"
    return (
        f"crop={geo.canvas_w}:{geo.visible_h}:0:{geo.bar_h},"
        f"pad={geo.canvas_w}:{geo.canvas_h}:0:{geo.bar_h}:color={pad_colour}"
    )
