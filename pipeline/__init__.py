"""Local YouTube voiceover post-production pipeline.

A modular, GPU-aware pipeline that turns a raw voiceover recording + script
into a finished, captioned, letterboxed video. Each pipeline stage lives in its
own module and is orchestrated by :mod:`pipeline.runner`.
"""

__version__ = "0.1.0"
