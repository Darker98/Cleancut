"""Stub of ``deepspeed.runtime.engine`` (inference-only).

``resemble_enhance.utils.engine.Engine`` subclasses ``DeepSpeedEngine``, so this
must be a real, subclassable class. It is never *instantiated* during
enhancement/inference, so its constructor refuses to run.
"""


class DeepSpeedEngine:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "deepspeed stub: DeepSpeedEngine cannot be instantiated. Only "
            "resemble-enhance inference (a plain torch forward pass) is "
            "supported on this install — training is not available."
        )
