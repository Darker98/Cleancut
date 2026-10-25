"""Stage 3 — Stutter / false-start detection.

Two passes:

1. **Heuristic** — over the word-timestamp JSON, find near-duplicate n-grams
   within a short rolling time window (fuzzy match via rapidfuzz) plus explicit
   restart phrases.
2. **Verification** — each candidate + surrounding context is sent to Groq
   (llama-3.3-70b-versatile) which classifies it as a genuine false-start vs.
   intentional repetition and proposes a precise cut range, returned as JSON.

Nothing is cut here. Candidates are surfaced for explicit user review.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from .config import StutterConfig
from .utils import PipelineError, probe_duration, run_ffmpeg, write_json

log = logging.getLogger(__name__)

_PUNCT = re.compile(r"[^\w']+")


def _norm(word: str) -> str:
    return _PUNCT.sub("", word.lower()).strip()


def _ngram_text(words: list[dict], i: int, n: int) -> str:
    return " ".join(_norm(words[j]["word"]) for j in range(i, i + n))


# --------------------------------------------------------------------------- #
# Pass 1 — heuristics
# --------------------------------------------------------------------------- #
def _detect_repeated_ngrams(words: list[dict], cfg: StutterConfig) -> list[dict]:
    """Find a phrase immediately (re)started within ``window_seconds``."""
    from rapidfuzz import fuzz

    candidates: list[dict] = []
    n_words = len(words)
    claimed: set[int] = set()  # word indices already part of a candidate

    for n in range(cfg.ngram_max, cfg.ngram_min - 1, -1):
        for i in range(0, n_words - 2 * n + 1):
            if any(idx in claimed for idx in range(i, i + n)):
                continue
            first = _ngram_text(words, i, n)
            if not first.strip():
                continue
            j = i + n
            # second occurrence must follow closely in time
            if words[j]["start"] - words[i + n - 1]["end"] > cfg.window_seconds:
                continue
            second = _ngram_text(words, j, n)
            score = fuzz.ratio(first, second)
            if score >= cfg.fuzzy_threshold:
                # The *first* occurrence is the false start to remove.
                cand = _make_candidate(
                    words, i, i + n - 1,
                    kind="repeated_phrase",
                    reason=f"Phrase '{first}' repeated (~{int(score)}% match) within "
                           f"{cfg.window_seconds:.0f}s.",
                )
                candidates.append(cand)
                claimed.update(range(i, i + n))
    return candidates


def _detect_restart_phrases(words: list[dict], cfg: StutterConfig) -> list[dict]:
    """Find explicit restart/correction phrases like 'let me start over'."""
    candidates: list[dict] = []
    norm_words = [_norm(w["word"]) for w in words]
    joined = " ".join(norm_words)

    # Map character offsets back to word indices.
    offsets: list[tuple[int, int]] = []
    pos = 0
    for w in norm_words:
        offsets.append((pos, pos + len(w)))
        pos += len(w) + 1

    for phrase in cfg.restart_phrases:
        for m in re.finditer(rf"(?<!\w){re.escape(phrase)}(?!\w)", joined):
            start_idx = _char_to_word(offsets, m.start())
            end_idx = _char_to_word(offsets, m.end() - 1)
            if start_idx is None or end_idx is None:
                continue
            candidates.append(_make_candidate(
                words, start_idx, end_idx,
                kind="restart_phrase",
                reason=f"Restart phrase detected: '{phrase}'.",
            ))
    return candidates


def _char_to_word(offsets: list[tuple[int, int]], char_pos: int) -> int | None:
    for idx, (a, b) in enumerate(offsets):
        if a <= char_pos <= b:
            return idx
    return None


def _make_candidate(words: list[dict], start_idx: int, end_idx: int,
                    kind: str, reason: str) -> dict:
    text = " ".join(words[k]["word"] for k in range(start_idx, end_idx + 1))
    return {
        "id": f"{kind}_{start_idx}_{end_idx}",
        "kind": kind,
        "start_word_idx": start_idx,
        "end_word_idx": end_idx,
        "start_time": words[start_idx]["start"],
        "end_time": words[end_idx]["end"],
        "text": text,
        "heuristic_reason": reason,
    }


def _dedupe(cands: list[dict]) -> list[dict]:
    """Drop overlapping candidates, keeping the longer span."""
    cands = sorted(cands, key=lambda c: (c["start_word_idx"],
                                         -(c["end_word_idx"] - c["start_word_idx"])))
    kept: list[dict] = []
    for c in cands:
        overlap = any(not (c["end_word_idx"] < k["start_word_idx"]
                           or c["start_word_idx"] > k["end_word_idx"]) for k in kept)
        if not overlap:
            kept.append(c)
    return sorted(kept, key=lambda c: c["start_word_idx"])


# --------------------------------------------------------------------------- #
# Pass 2 — Groq verification
# --------------------------------------------------------------------------- #
_GROQ_SYSTEM = (
    "You are an audio editing assistant. You are given a transcript span that a "
    "heuristic flagged as a possible false start / stutter (the speaker restarting "
    "a sentence), with surrounding context. Decide whether the FLAGGED span is a "
    "genuine false start that should be cut, versus an intentional repetition, "
    "emphasis, or normal speech. Respond ONLY with JSON of the form: "
    '{"is_false_start": true|false, "cut_start_word_idx": int, '
    '"cut_end_word_idx": int, "reasoning": "short explanation"}. '
    "The cut indices must lie within the flagged span's index range and should "
    "tightly bound only the words to remove."
)


def _build_context(words: list[dict], cand: dict, ctx: int) -> str:
    lo = max(0, cand["start_word_idx"] - ctx)
    hi = min(len(words) - 1, cand["end_word_idx"] + ctx)
    parts = []
    for k in range(lo, hi + 1):
        token = words[k]["word"]
        if k == cand["start_word_idx"]:
            token = "[[FLAG_START]] " + token
        if k == cand["end_word_idx"]:
            token = token + " [[FLAG_END]]"
        parts.append(f"({k}){token}")
    return " ".join(parts)


def _verify_with_groq(words: list[dict], candidates: list[dict],
                      cfg: StutterConfig, api_key: str) -> None:
    try:
        from groq import Groq
    except Exception as exc:  # pragma: no cover
        raise PipelineError(
            "The `groq` package is not installed. `pip install groq`. "
            f"Underlying error: {exc}"
        ) from exc

    client = Groq(api_key=api_key)
    for cand in candidates:
        context = _build_context(words, cand, cfg.groq_context_words)
        user = (
            f"Flagged span words {cand['start_word_idx']}–{cand['end_word_idx']}: "
            f"\"{cand['text']}\"\nReason flagged: {cand['heuristic_reason']}\n"
            f"Context (each token tagged with its word index):\n{context}"
        )
        try:
            resp = client.chat.completions.create(
                model=cfg.groq_model,
                messages=[
                    {"role": "system", "content": _GROQ_SYSTEM},
                    {"role": "user", "content": user},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content)
            cand["groq"] = {
                "is_false_start": bool(data.get("is_false_start", False)),
                "cut_start_word_idx": int(data.get("cut_start_word_idx",
                                                    cand["start_word_idx"])),
                "cut_end_word_idx": int(data.get("cut_end_word_idx",
                                                  cand["end_word_idx"])),
                "reasoning": str(data.get("reasoning", "")),
            }
        except Exception as exc:  # network / parse failure → keep candidate, flag it
            log.warning("Groq verification failed for %s: %s", cand["id"], exc)
            cand["groq"] = {
                "is_false_start": None,
                "cut_start_word_idx": cand["start_word_idx"],
                "cut_end_word_idx": cand["end_word_idx"],
                "reasoning": f"Groq verification unavailable ({exc}).",
            }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def detect_stutters(transcript: dict, out_json: Path, cfg: StutterConfig,
                    groq_api_key: str | None, progress=None) -> list[dict]:
    """Return a list of review-ready candidate cuts and persist them to JSON."""
    words = transcript["words"]
    if progress:
        progress(0.1, "Heuristic scan")

    candidates = _detect_repeated_ngrams(words, cfg)
    candidates += _detect_restart_phrases(words, cfg)
    candidates = _dedupe(candidates)
    log.info("Stage 3: %d heuristic candidates", len(candidates))

    if cfg.use_groq and candidates:
        if not groq_api_key:
            log.warning("GROQ_API_KEY not set — skipping verification, surfacing "
                        "heuristic candidates only.")
            for c in candidates:
                c["groq"] = {
                    "is_false_start": None,
                    "cut_start_word_idx": c["start_word_idx"],
                    "cut_end_word_idx": c["end_word_idx"],
                    "reasoning": "GROQ_API_KEY not set; not verified.",
                }
        else:
            if progress:
                progress(0.4, f"Verifying {len(candidates)} candidates with Groq")
            _verify_with_groq(words, candidates, cfg, groq_api_key)

    # Default the user-facing decision: pre-accept confirmed false starts.
    for c in candidates:
        groq = c.get("groq", {})
        c["suggested_accept"] = bool(groq.get("is_false_start"))
        c["accepted"] = c["suggested_accept"]
        # Effective cut range = Groq's tightened range when available.
        c["cut_start_word_idx"] = groq.get("cut_start_word_idx", c["start_word_idx"])
        c["cut_end_word_idx"] = groq.get("cut_end_word_idx", c["end_word_idx"])

    write_json(out_json, candidates)
    if progress:
        progress(1.0, f"{len(candidates)} candidates flagged")
    return candidates


# --------------------------------------------------------------------------- #
# Applying accepted cuts
# --------------------------------------------------------------------------- #
def _accepted_spans(transcript: dict, candidates: list[dict]) -> list[tuple[float, float]]:
    """Convert accepted candidates into merged (start, end) time spans to remove."""
    words = transcript["words"]
    spans: list[tuple[float, float]] = []
    for c in candidates:
        if not c.get("accepted"):
            continue
        a = max(0, int(c["cut_start_word_idx"]))
        b = min(len(words) - 1, int(c["cut_end_word_idx"]))
        if b < a:
            continue
        spans.append((words[a]["start"], words[b]["end"]))
    spans.sort()
    # Merge overlaps.
    merged: list[tuple[float, float]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1] + 1e-3:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def apply_accepted_cuts(audio_path: Path, transcript: dict, candidates: list[dict],
                        out_path: Path) -> Path:
    """Render a new audio file with the accepted false-start spans removed.

    If nothing is accepted, the input is simply copied to ``out_path``.
    """
    audio_path = Path(audio_path)
    out_path = Path(out_path)
    spans = _accepted_spans(transcript, candidates)

    if not spans:
        log.info("No accepted stutter cuts; copying audio unchanged.")
        run_ffmpeg(["-i", str(audio_path), "-c", "copy", str(out_path)],
                   desc="copy audio (no cuts)")
        return out_path

    duration = probe_duration(audio_path)
    # Build the complement: the kept segments between removed spans.
    keep: list[tuple[float, float]] = []
    cursor = 0.0
    for s, e in spans:
        if s > cursor:
            keep.append((cursor, s))
        cursor = max(cursor, e)
    if cursor < duration:
        keep.append((cursor, duration))

    # ffmpeg atrim + concat over kept segments.
    parts = []
    labels = []
    for i, (s, e) in enumerate(keep):
        parts.append(
            f"[0:a]atrim=start={s:.3f}:end={e:.3f},asetpts=PTS-STARTPTS[a{i}]"
        )
        labels.append(f"[a{i}]")
    filter_complex = ";".join(parts) + ";" + "".join(labels) + \
        f"concat=n={len(keep)}:v=0:a=1[out]"

    run_ffmpeg(
        ["-i", str(audio_path), "-filter_complex", filter_complex,
         "-map", "[out]", str(out_path)],
        desc="apply stutter cuts",
    )
    log.info("Applied %d stutter cut span(s) -> %s", len(spans), out_path)
    return out_path
