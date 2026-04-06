"""torch.compile only when Inductor can use Triton; else return eager model (HPC-safe)."""

import torch


def maybe_compile(model, *, device, enabled: bool, mode: str = "default"):
    """Returns (model, status) where status is logged to wandb."""
    if not enabled or getattr(device, "type", str(device)) != "cuda":
        return model, "off"
    try:
        import triton  # noqa: F401 — GPU inductor kernels need it
    except ImportError:
        return model, "skipped_no_triton"
    try:
        return torch.compile(model, mode=mode), f"on_{mode}"
    except Exception:
        return model, "skipped_compile_error"
