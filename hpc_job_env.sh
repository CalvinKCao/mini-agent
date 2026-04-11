# Sourced by all Slurm batch scripts.
# In a Slurm job: builds a fresh venv on $SLURM_TMPDIR (fast node-local SSD)
# using Alliance Canada's pre-built wheel cache (--no-index).  Imports from
# local disk are orders of magnitude faster than reading thousands of small
# .so files from /scratch over the parallel filesystem.
# On a login node: falls back to the persistent .venv under PROJECT_ROOT.

_resolve_root() {
    local d
    for d in \
        "${SLURM_SUBMIT_DIR:-}" \
        "${SCRATCH:-}/mini-agent" \
        "${SCRATCH:-}/overcooked" \
        "${HOME}/mini-agent" \
        "${HOME}/overcooked"
    do
        [ -z "$d" ] && continue
        d=$(readlink -f "$d" 2>/dev/null) || continue
        if [ -f "$d/train.py" ] || [ -f "$d/train_v2.py" ]; then
            echo "$d"
            return 0
        fi
    done
    return 1
}

if ! PROJECT_ROOT=$(_resolve_root); then
    echo "ERROR: could not find repo root (need train.py or train_v2.py)."
    exit 1
fi

mkdir -p "$PROJECT_ROOT/logs"

module purge || true
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load cudnn/8.9

# Resolve $PROJECT for persistent storage (checkpoints, wandb)
if [ -z "${PROJECT:-}" ] && [ -d "$HOME/projects" ]; then
    shopt -s nullglob
    _proj_matches=("$HOME"/projects/def-* "$HOME"/projects/aip-*)
    shopt -u nullglob
    if [ "${#_proj_matches[@]}" -gt 0 ]; then
        export PROJECT=$(readlink -f "${_proj_matches[0]}")
    fi
fi

# ── venv ─────────────────────────────────────────────────────────────
if [ -n "${SLURM_TMPDIR:-}" ]; then
    # In a Slurm job: build a fresh venv on the compute node's local SSD.
    # Alliance CA's wheel cache (--no-index) makes this fast and avoids
    # hammering the parallel filesystem with thousands of small reads.
    VENV_DIR="$SLURM_TMPDIR/env"
    echo "Building node-local venv at $VENV_DIR ..."
    virtualenv --no-download "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --no-index --upgrade pip -q

    # Heavy C-extension packages — from Alliance CA's pre-built wheels, no download
    pip install --no-index torch numpy scikit-learn -q

    # triton enables torch.compile; optional — compile_safe.py falls back gracefully
    pip install --no-index triton -q 2>/dev/null \
        || pip install triton -q 2>/dev/null \
        || true

    # wandb isn't in the Alliance CA wheelhouse; pull from PyPI (pure-Python, small)
    pip install wandb -q

else
    # Login node / local dev: use the persistent venv on disk
    VENV_DIR="${PROJECT_ROOT}/.venv"
    if [ ! -d "$VENV_DIR" ]; then
        echo "Creating venv at $VENV_DIR ..."
        python -m venv "$VENV_DIR"
        source "$VENV_DIR/bin/activate"
        pip install --upgrade pip -q
        pip install torch --index-url https://download.pytorch.org/whl/cu121 -q
        pip install numpy wandb scikit-learn -q
    else
        source "$VENV_DIR/bin/activate"
        python -c "import sklearn" 2>/dev/null || pip install scikit-learn -q
        python -c "import triton"  2>/dev/null || pip install triton -q || true
    fi
fi

cd "$PROJECT_ROOT" || exit 1
