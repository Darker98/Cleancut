# 🎬 Cleancut

A local voiceover post-production pipeline. Turn a raw voiceover recording + script into a finished, captioned, cinematic
YouTube video — **entirely locally and free** (the only network call is to
Groq's free tier for stutter verification).

The pipeline:

1. **Prepare** audio — extract the **original voice track unchanged** (no
   enhancement/denoising; just decoded to wav)
2. **Transcribe** with word-level timestamps (faster-whisper)
3. **Detect stutters / false starts** (rapidfuzz heuristics + Groq verification) — *you review before anything is cut*
4. **Trim pauses / dead air** (auto-editor)
5. **Assemble background clips** timed to ~60s / `[BREAK]` segments, extending short clips with an invisible reverse-loop, joined by a soft **2s crossfade** between clips
6. **Generate styled captions** (`.ass`, bold sans-serif, white text on a 50%-opaque background box) — **editable in the UI**
7. **Composite** audio + video + captions
8. **Letterbox** to a 2.4:1 cinematic frame

Everything runs through a single local **Gradio** web app with live progress feedback.

---

## Hardware notes (designed for 6 GB VRAM)

Built and tuned for a laptop **RTX 4050 (~6 GB VRAM)** on **Windows**:

- Whisper defaults to `large-v3` at **`int8_float16`** (fits in 6 GB). If you hit
  out-of-memory, switch (in the Settings panel) to **`distil-large-v3`** or
  **`medium`**, and/or compute type **`int8`**.
- Voice enhancement is **disabled** — the original recording is kept as-is, so
  the only GPU-heavy stage is transcription. (resemble-enhance is still
  installed but unused; you can skip it on a fresh setup if you like.)
- The pipeline calls `torch.cuda.empty_cache()` after transcription to release
  VRAM before later stages.
- If CUDA is unavailable, transcription falls back to CPU (slow but functional).

---

## Setup (Windows)

### 1. Python 3.11

`torch` and `resemble-enhance` do **not** ship wheels for Python 3.12+. Install
[Python 3.11](https://www.python.org/downloads/release/python-3119/) and create
the venv with it explicitly:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
python --version   # should print 3.11.x
```

### 2. Run the setup script

```powershell
.\scripts\setup.ps1
```

This installs everything **in the correct order** and verifies the imports at
the end. It exists because a naive `pip install -r requirements.txt` fails on
Windows for three reasons the script handles:

1. **numpy** must be `1.26.x` — torch 2.1.1 is incompatible with numpy 2.x
   (the "Failed to initialize NumPy" error).
2. **setuptools** must be `< 70` — 70+ removed `pkg_resources.packaging`, which
   torch 2.1.1 still imports.
3. **deepspeed** (a hard dependency of resemble-enhance, pinned to `0.12.4`)
   cannot build on Windows without the full CUDA *toolkit* (nvcc) + a C++
   compiler. But resemble-enhance only uses deepspeed for *training*; its
   enhancement path is a plain torch forward pass. So we install a tiny
   **inference-only deepspeed stub** (`stubs/deepspeed-stub/`) that satisfies
   the dependency, and install `resemble-enhance` with `--no-deps`. No CUDA
   toolkit or compiler required.

The script installs CUDA-enabled torch from the **`cu121`** wheel index, which
is compatible with your driver (CUDA UMD 13.3). Verify CUDA afterwards:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
# expected: 2.1.1+cu121 True 12.1
```

> If `cuda.is_available()` is `False`, edit `scripts/setup.ps1` to use the
> `cu118` index URL instead, or update the NVIDIA driver. The app prints the
> detected GPU/CUDA summary at the top of the page.

> **auto-editor** downloads a small platform binary on its first run, so the
> very first pause-trim needs internet once (cached thereafter).

### 3. ffmpeg (with libass)

Captions are burned via libass, so you need a **full** ffmpeg build:

```powershell
winget install Gyan.FFmpeg
# or: choco install ffmpeg-full
```

Then restart the terminal and confirm:

```powershell
ffmpeg -version | findstr libass
```

The app refuses to start a render if `libass` is missing.

### 4. Caption font

Download **Montserrat Bold** and save it as
`assets/fonts/Montserrat-Bold.ttf` — see [`assets/fonts/README.md`](assets/fonts/README.md).

### 5. Groq API key

```powershell
copy .env.example .env
# edit .env and set GROQ_API_KEY=...
```

Get a free key at <https://console.groq.com>. Without it, Stage 3 still flags
candidates heuristically — they're just shown as "unverified" for your review.

---

## Run

```powershell
.venv\Scripts\activate
python app.py
```

This opens `http://127.0.0.1:7860` in your browser.

### Flow

1. **Inputs** — upload the voiceover (an audio file *or* a video like mp4/mkv —
   the audio track is extracted automatically), paste the script, set the clips
   folder path.
2. **Settings** — tune Whisper model/VRAM, pause sensitivity, caption limits,
   segment length, aspect ratio, clip order, **clip crossfade duration**, and
   **caption background opacity**.
3. **Run analysis (Stages 1–3)** — prepare original audio, transcribe, flag
   stutters. Progress streams live (the slow step is transcription).
4. **Review flagged stutters** — each card plays just that span (with waveform);
   Accept / Reject / adjust the cut word range. Nothing is cut without your OK.
5. **Continue (Stages 4–8)** — applies accepted cuts, trims pauses, re-transcribes
   the cleaned audio, assembles background (with crossfades), captions,
   composites, and letterboxes. Progress streams live.
6. **Edit captions** — after the first render, the caption events appear in an
   editable table. Fix any typo or timing and click **Re-render captions** — it
   only redoes the caption + composite pass (fast; no re-transcription or
   background rebuild).
7. **Preview & download** the final `.mp4`.

---

## Folder conventions

### Background clips

Point the app at a folder of video clips (no audio needed — the voiceover is the
audio track):

```
D:\my-clips\
  01-intro.mp4
  02-broll.mov
  03-cityscape.mp4
```

Supported: `.mp4 .mov .mkv .webm .avi .m4v .wmv .flv`. In **sequential** mode
clips are used in filename order and cycled; in **random** mode the order is
shuffled deterministically. Clips shorter than a segment are extended with a
reverse-loop ping-pong (clip → reversed → clip …) trimmed to the exact length,
so the loop seam is invisible.

### Script `[BREAK]` markers

Paste your script and insert `[BREAK]` wherever the background video should
switch topics:

```
Welcome back to the channel. Today we're talking about latency. [BREAK]
First, let's look at where the time actually goes in a request. [BREAK]
Now here's the part everyone gets wrong...
```

Each background segment runs until the next `[BREAK]` or `segment_seconds`
(default 60s), whichever comes first. Markers are aligned to the **post-cut**
transcript, so they track the real edited timeline.

---

## Caption style

Phrase-level chunks (≈3–7 words), single line each (auto-split if too wide),
≤2.4s on screen, sentence case, pure white text on a **semi-opaque background
box** (50% by default — set the opacity to 0 in Settings for outline-only),
centered, sized to ~6.5% of the **visible** (non-letterboxed) frame height,
sitting ~9% up from the visible-frame bottom — never inside the black bars.
All tunable in `pipeline/config.py` (`CaptionConfig`), and the caption **text /
timing is editable in the UI** after the first render.

---

## Project structure

```
app.py                     # Gradio frontend
pipeline/
  config.py                # all tunables (dataclasses)
  gpu.py                   # CUDA detection + VRAM release helpers
  utils.py                 # ffmpeg/ffprobe wrappers, JSON IO
  align.py                 # script [BREAK] ↔ transcript alignment
  runner.py                # orchestrates the two phases
  stage1_prepare.py        # decode original audio (no enhancement)
  stage2_transcribe.py     # faster-whisper (word timestamps)
  stage3_stutter.py        # heuristic + Groq detection, + cut application
  stage4_pauses.py         # auto-editor
  stage5_background.py     # clip assembly + reverse-loop
  stage6_captions.py       # .ass generation
  stage7_composite.py      # final composite + letterbox render
  stage8_letterbox.py      # letterbox geometry (shared with captions)
assets/fonts/              # bundled caption font
stubs/deepspeed-stub/      # inference-only deepspeed stub (Windows install fix)
scripts/setup.ps1          # ordered one-shot dependency install
work/<run_id>/             # per-run intermediates + final.mp4
requirements.txt
.env.example
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDA out of memory` loading Whisper | Settings → smaller model (`distil-large-v3`/`medium`) or compute type `int8`. |
| `ffmpeg ... does not include libass` | Install a full ffmpeg build (gyan.dev / `winget install Gyan.FFmpeg`). |
| `auto-editor not found` | `pip install auto-editor` inside the venv. |
| Captions render in the wrong font | Confirm `assets/fonts/Montserrat-Bold.ttf` exists and its family name matches `CaptionConfig.font_name`. |
| `torch.cuda.is_available()` is False | Reinstall torch from the matching CUDA wheel index (`cu121`/`cu118`), or update the NVIDIA driver. |
| `Failed to initialize NumPy` | numpy got upgraded to 2.x; run `pip install "numpy==1.26.4"`. |
| deepspeed build error / `CUDA_HOME does not exist` | You ran a raw `pip install`; use `scripts/setup.ps1`, which installs the deepspeed stub instead of building it. |
| `cannot import name 'HfFolder'` (gradio) | `pip install "huggingface_hub==0.25.2"`. |
| App launches then `500` / `unhashable type: 'dict'` | fastapi/starlette too new for gradio 4.44.1; `pip install "fastapi==0.112.2" "starlette==0.38.6"`. |
| App `500` / `argument of type 'bool' is not iterable` | gradio_client schema bug; `app.py` patches it at startup — run via `python app.py`, don't bypass it. |
| `When localhost is not accessible…` | A symptom of the two rows above (the 500 fails the health check); fix those and it clears. |
| Groq errors / no key | Set `GROQ_API_KEY` in `.env`; candidates still surface as "unverified" without it. |
| Groq: `unexpected keyword argument 'proxies'` | httpx too new for groq 0.11.0; `pip install "httpx==0.27.2"`. |
| Background clips folder empty | Add supported video files; the app validates this before rendering. |

---

## Non-negotiables honoured

- No destructive auto-cutting — stutter removal requires explicit per-candidate approval.
- No paid APIs except Groq (free tier).
- Fully offline except the Groq verification calls.
- All GPU-heavy stages designed around the 6 GB VRAM ceiling (sequential loading,
  quantized compute, chunked processing, explicit cache release).
