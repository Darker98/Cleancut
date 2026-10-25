"""Stub of ``deepspeed.accelerator`` (inference-only)."""


class _Accelerator:
    def communication_backend_name(self) -> str:
        return "nccl"

    def __getattr__(self, name):
        # Any other accelerator method is a no-op; only reached on training.
        def _noop(*args, **kwargs):
            return None
        return _noop


def get_accelerator() -> "_Accelerator":
    return _Accelerator()
