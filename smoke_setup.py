#!/usr/bin/env python3
"""Smoke-test repo setup before long Slurm jobs.

Two modes:
  * Default (CPU / login-node safe): stdlib + numpy + torch import, env stepping,
    optional sklearn/wandb imports. No CUDA required.
  * --gpu: needs a GPU (compute node). DRC forward, optional torch.compile via compile_safe.

Examples:
  # On login node (Alliance: keep it quick, ~CPU only):
  cd $SCRATCH/mini-agent && source .venv/bin/activate && python smoke_setup.py

  # After interactive GPU allocation:
  salloc --account=YOURGROUP --gres=gpu:l40s:1 --cpus-per-task=2 --mem=8G --time=0:20:0
  module load ...; source .venv/bin/activate; cd $SCRATCH/mini-agent; python smoke_setup.py --gpu

  # Queued smoke job (~10–15 min wall, catches Slurm+modules+venv+GPU):
  sbatch slurm_smoke.sh
"""

from __future__ import annotations

import argparse
import os
import sys


def _ok(msg: str) -> None:
    print(f"  OK  {msg}")


def _fail(msg: str) -> None:
    print(f"  FAIL {msg}", file=sys.stderr)


def phase_imports() -> bool:
    print("[1] Core imports")
    try:
        import numpy as np  # noqa: F401
        _ok("numpy")
    except Exception as e:
        _fail(f"numpy: {e}")
        return False

    try:
        import torch
        _ok(f"torch {torch.__version__}")
    except Exception as e:
        _fail(f"torch: {e}")
        return False

    try:
        import wandb
        _ok(f"wandb {wandb.__version__}")
    except Exception as e:
        _fail(f"wandb: {e}")
        return False

    try:
        import sklearn  # noqa: F401
        _ok("sklearn (interpretability.py)")
    except Exception as e:
        print(f"  WARN sklearn (optional): {e}")

    try:
        from env import ForkedCorridorVec
        _ok("env.ForkedCorridorVec")
    except Exception as e:
        _fail(f"env: {e}")
        return False

    try:
        from models_drc import DRCActorCritic
        _ok("models_drc.DRCActorCritic")
    except Exception as e:
        _fail(f"models_drc: {e}")
        return False

    try:
        from compile_safe import maybe_compile
        _ok("compile_safe.maybe_compile")
    except Exception as e:
        _fail(f"compile_safe: {e}")
        return False

    return True


def phase_env_cpu() -> bool:
    print("[2] Vectorized env (CPU, few steps)")
    try:
        from env import ForkedCorridorVec
        import numpy as np

        e = ForkedCorridorVec(num_envs=4, max_steps=20, seed=0)
        obs, gv = e.reset()
        assert obs.shape == (4, 2, 3, 7, 7)
        assert gv.shape == (4, 2, 2)
        for _ in range(5):
            a = np.random.randint(0, 5, size=(4, 2))
            obs, gv, r, d, info = e.step(a)
        _ok("ForkedCorridorVec reset + 5 steps")
    except Exception as ex:
        _fail(str(ex))
        return False
    return True


def phase_gpu(compile_smoke: bool) -> bool:
    print("[3] CUDA + DRC forward")
    import torch
    from models_drc import DRCActorCritic
    from compile_safe import maybe_compile

    if not torch.cuda.is_available():
        _fail("torch.cuda.is_available() is False (need a GPU node)")
        return False
    dev = torch.device("cuda")
    _ok(f"device {torch.cuda.get_device_name(0)}")

    try:
        t = torch.randn(4096, device=dev)
        assert float((t * t).sum()) >= 0.0
        _ok("tiny GPU tensor op")
    except Exception as e:
        _fail(f"GPU tensor op: {e}")
        return False

    try:
        m = DRCActorCritic(G=32, D=3, N=3).to(dev)
        h, c = m.initial_state(8, device=dev)
        obs = torch.randn(8, 3, 7, 7, device=dev)
        goal = torch.zeros(8, 2, device=dev)
        goal[:, 0] = 1.0
        with torch.no_grad():
            logits, v, h2, c2 = m.forward_logits(obs, goal, h, c)
        assert logits.shape == (8, 5)
        assert v.shape == (8,)
        _ok("DRCActorCritic.forward_logits (eager)")
    except Exception as e:
        _fail(f"DRC forward: {e}")
        return False

    if compile_smoke:
        print("[4] torch.compile (optional, needs Triton)")
        try:
            m2 = DRCActorCritic(G=32, D=3, N=3).to(dev)
            m2, st = maybe_compile(m2, device=dev, enabled=True, mode="default")
            h, c = m2.initial_state(8, device=dev)
            with torch.no_grad():
                m2.forward_logits(obs, goal, h, c)
            _ok(f"compiled path status={st}")
        except Exception as e:
            _fail(f"compile smoke: {e}")
            return False

    return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--gpu",
        action="store_true",
        help="Run CUDA + DRC checks (run on a compute node with a GPU)",
    )
    p.add_argument(
        "--compile-smoke",
        action="store_true",
        help="With --gpu, also run maybe_compile + one forward (needs Triton)",
    )
    args = p.parse_args()

    print("smoke_setup.py — cwd:", os.getcwd())
    print("python:", sys.executable)

    if not phase_imports():
        return 1
    if not phase_env_cpu():
        return 1

    if args.gpu:
        if not phase_gpu(args.compile_smoke):
            return 1
    else:
        print("[3] Skipped GPU checks (pass --gpu on a GPU node or use sbatch slurm_smoke.sh)")

    print("\nAll smoke checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
