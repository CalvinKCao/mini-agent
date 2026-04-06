"""torch.compile wrapper with graceful fallback."""
import torch


def maybe_compile(model, device, enabled=True, mode="default"):
    """Try torch.compile; fall back to eager if unavailable or compile fails.

    Returns (model, status_str) where status_str is one of:
      'on_<mode>'  — compiled successfully
      'off'        — compile disabled or not on CUDA
      'error'      — compile failed, using eager
    """
    dev_type = device.type if hasattr(device, "type") else str(device)
    if not enabled or dev_type != "cuda":
        return model, "off"
    try:
        compiled = torch.compile(model, mode=mode)
        return compiled, f"on_{mode}"
    except Exception:
        return model, "error"
