"""Script ↔ transcript alignment.

The recording follows the script in order, so a simple ordered alignment of the
normalised word sequences (via :class:`difflib.SequenceMatcher`) is enough to
convert the script's ``[BREAK]`` markers into real timestamps on the final
(post-cut) audio timeline.
"""
from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

BREAK_TOKEN = "[BREAK]"
_PUNCT = re.compile(r"[^\w']+")


def _norm(word: str) -> str:
    return _PUNCT.sub("", word.lower()).strip()


def parse_script(script: str) -> tuple[list[str], list[int]]:
    """Return (normalised script words, break positions).

    A break position ``p`` means a ``[BREAK]`` marker occurs *before* script
    word ``p`` (i.e. a topic change starts at that word).
    """
    raw_tokens = script.replace(BREAK_TOKEN, f" {BREAK_TOKEN} ").split()
    words: list[str] = []
    breaks: list[int] = []
    for tok in raw_tokens:
        if tok == BREAK_TOKEN:
            breaks.append(len(words))
        else:
            n = _norm(tok)
            if n:
                words.append(n)
    # De-dupe consecutive/edge breaks.
    breaks = sorted({b for b in breaks if 0 < b < len(words)})
    return words, breaks


def _script_to_transcript_map(script_words: list[str],
                              transcript_words: list[dict]) -> dict[int, int]:
    """Map script-word index → transcript-word index using matching blocks."""
    from difflib import SequenceMatcher

    t_norm = [_norm(w["word"]) for w in transcript_words]
    sm = SequenceMatcher(a=script_words, b=t_norm, autojunk=False)
    mapping: dict[int, int] = {}
    for block in sm.get_matching_blocks():
        for k in range(block.size):
            mapping[block.a + k] = block.b + k
    return mapping


def align_breaks(script: str, transcript: dict) -> list[float]:
    """Return sorted break timestamps (seconds) on the transcript timeline."""
    script_words, break_positions = parse_script(script)
    transcript_words = transcript["words"]
    duration = transcript.get("duration") or (
        transcript_words[-1]["end"] if transcript_words else 0.0
    )
    if not break_positions or not transcript_words:
        return []

    mapping = _script_to_transcript_map(script_words, transcript_words)
    times: list[float] = []
    for p in break_positions:
        # Find the nearest mapped script word at/after the break position.
        t_idx = None
        for q in range(p, len(script_words)):
            if q in mapping:
                t_idx = mapping[q]
                break
        if t_idx is None:  # break falls past the last matched word
            t_idx = len(transcript_words) - 1
        t = transcript_words[t_idx]["start"]
        if 0.5 < t < duration - 0.5:
            times.append(round(t, 3))

    times = sorted(set(times))
    log.info("Aligned %d break marker(s) -> %s", len(times), times)
    return times


def build_segments(break_times: list[float], total_duration: float,
                   segment_seconds: float) -> list[tuple[float, float]]:
    """Build background segments: each ≤ segment_seconds, split at break times.

    A segment ends at whichever comes first — the next break or the
    ``segment_seconds`` cap from its start.
    """
    boundaries = sorted({0.0, total_duration, *[
        b for b in break_times if 0.0 < b < total_duration
    ]})
    segments: list[tuple[float, float]] = []
    for i in range(len(boundaries) - 1):
        start, hard_end = boundaries[i], boundaries[i + 1]
        cursor = start
        while hard_end - cursor > 1e-3:
            seg_end = min(cursor + segment_seconds, hard_end)
            segments.append((round(cursor, 3), round(seg_end, 3)))
            cursor = seg_end
    return segments
