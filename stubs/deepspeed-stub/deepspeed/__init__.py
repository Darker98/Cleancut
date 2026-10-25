"""Inference-only stub of `deepspeed` (version 0.12.4).

resemble-enhance hard-depends on ``deepspeed==0.12.4`` and imports a handful of
its symbols at module-import time (notably ``class Engine(DeepSpeedEngine)`` in
its training utilities). The real deepspeed cannot be pip-installed on Windows
without the CUDA toolkit + a C++ compiler.

But the *enhancement / inference* path in resemble-enhance is a plain torch
``nn.Module`` forward pass — it never instantiates the deepspeed engine or calls
any deepspeed function. So we only need these names to *exist* so the imports
succeed. Every callable here raises if actually invoked, which only happens on
the training path we never use.
"""
from . import accelerator, runtime  # noqa: F401  (attach submodules)

__version__ = "0.12.4"

_TRAINING_ONLY = (
    "This is the inference-only deepspeed stub bundled with the voiceover "
    "pipeline. The deepspeed training engine is not available — only "
    "resemble-enhance *inference* is supported on this install."
)


class DeepSpeedConfig:  # imported by resemble_enhance.enhancer.train
    def __init__(self, *args, **kwargs):
        # Constructed only in training config setup; tolerate construction so a
        # stray import-time instantiation won't explode, but hold no state.
        self._args = args
        self._kwargs = kwargs


def init_distributed(*args, **kwargs):
    raise RuntimeError(_TRAINING_ONLY)


def initialize(*args, **kwargs):
    raise RuntimeError(_TRAINING_ONLY)
