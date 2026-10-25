"""GPU helpers.

The 6 GB VRAM ceiling means GPU stages (enhancement, transcription) must run
sequentially and release memory between them. These helpers centralise that
discipline and provide friendly diagnostics.
"""
from __future__ import annotations

import gc
import logging

log = logging.getLogger(__name__)


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def device_string(prefer_cuda: bool = True) -> str:
    """Return 'cuda' if usable, else 'cpu'."""
    if prefer_cuda and cuda_available():
        return "cuda"
    return "cpu"


def describe_environment() -> str:
    """Human-readable summary used at startup / in error messages."""
    if not torch_available():
        return "PyTorch not installed."
    import torch

    lines = [f"torch {torch.__version__}"]
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        total = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
        lines.append(f"CUDA available: {name} ({total:.1f} GiB)")
        lines.append(f"CUDA runtime: {torch.version.cuda}")
    else:
        lines.append("CUDA NOT available — stages will fall back to CPU (slow).")
    return " | ".join(lines)


def free_gpu_memory() -> None:
    """Release cached GPU memory between sequential stages."""
    gc.collect()
    if not torch_available():
        return
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            log.debug("Released cached CUDA memory.")
    except Exception as exc:  # pragma: no cover - best effort
        log.debug("free_gpu_memory failed: %s", exc)


def is_oom_error(exc: BaseException) -> bool:
    """Best-effort detection of CUDA out-of-memory errors across libraries."""
    msg = str(exc).lower()
    needles = ("out of memory", "cuda oom", "cublas", "cudnn", "alloc", "cuda error")
    return any(n in msg for n in needles) and ("memory" in msg or "alloc" in msg or "oom" in msg)
