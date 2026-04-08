# Sourced by Slurm batch scripts (same cwd + modules + venv as training).
# Usage: SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); source "$SCRIPT_DIR/hpc_job_env.sh"

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
        if [ -f "$d/train_v2.py" ]; then
            echo "$d"
            return 0
        fi
    done
    return 1
}

if ! PROJECT_ROOT=$(_resolve_root); then
    echo "ERROR: could not find repo root (need train_v2.py)."
    exit 1
fi

mkdir -p "$PROJECT_ROOT/logs"

module purge || true
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load cudnn/8.9

if [ -z "${PROJECT:-}" ] && [ -d "$HOME/projects" ]; then
    shopt -s nullglob
    _proj_matches=("$HOME"/projects/def-* "$HOME"/projects/aip-*)
    shopt -u nullglob
    if [ "${#_proj_matches[@]}" -gt 0 ]; then
        export PROJECT=$(readlink -f "${_proj_matches[0]}")
    fi
fi

VENV_DIR="${PROJECT_ROOT}/.venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating venv at $VENV_DIR ..."
    python -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip -q
    pip install torch --index-url https://download.pytorch.org/whl/cu121 -q
    pip install triton -q || true
    pip install numpy wandb scikit-learn -q
else
    source "$VENV_DIR/bin/activate"
    if ! python -c "import triton" 2>/dev/null; then
        echo "Installing triton (optional, for torch.compile) ..."
        pip install triton -q || true
    fi
    if ! python -c "import sklearn" 2>/dev/null; then
        echo "Installing scikit-learn ..."
        pip install scikit-learn -q
    fi
fi

cd "$PROJECT_ROOT" || exit 1
