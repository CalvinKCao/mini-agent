---
name: alliancecan
description: Alliance Canada Slurm clusters — default to L40S on Killarney, jobs, storage, modules, paths, and account pitfalls. Use when editing Slurm scripts, cluster setup, or HPC workflows for docs.alliancecan.ca systems.
---

# Alliance Canada HPC

Apply when working on Slurm job scripts, cluster setup, paths, or GPU requests for Alliance national systems (Killarney, Fir, Narval, Nibi, Rorqual, Trillium, etc.).

## Default GPU for this repo: **L40S (always prefer this first)**

For **Killarney**, treat **NVIDIA L40S** as the **default** GPU in new or edited Slurm scripts unless the user explicitly asks for H100/A100 or you know they need 80 GB VRAM.

**Why:** L40S usually has a **much shorter queue** than H100 performance tiers. H100 is overkill for many PoC / IPPO / probe workloads and can sit in `PD` for hours.

### How to request L40S (standard pattern)

In the job script, use **`--gres`** — **do not** put H100-only partitions (`gpubase_h100_*`) together with L40S `gres` (incompatible).

```bash
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
# Optional: shorter wall for faster scheduling
#SBATCH --time=1-00:00:00
```

Submit from the repo directory:

```bash
sbatch --account=<your-ccdb-group> your_job.sh
# or if #SBATCH --account= is already in the file:
sbatch your_job.sh
```

**Verify** what Slurm recorded (after submit):

```bash
scontrol show job=<JOBID> | egrep -i 'ReqTRES|TRES=|MinMem'
```

You should see **`l40s`** in the GRES line, not `h100`.

### When to use H100 instead (opt-in)

Use **H100** only when the workload needs **large GPU memory** (e.g. huge batches, very large models) or the user asks for the performance tier. Typical Killarney pattern:

```bash
#SBATCH --partition=gpubase_h100_b4
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
```

Match **`b1`…`b5`** partition letter to **max wall time** allowed on that queue (`sinfo -o "%P %G %l"`). **Never** mix `gpubase_h100_*` with `--gres=gpu:l40s:1`.

---

## Slurm account (read this first — common failure mode)

- **Every job** needs `#SBATCH --account=<group-name>` where `<group-name>` is the **Group Name** from [CCDB](https://ccdb.alliancecan.ca/) → My Projects → My Resources and Allocations. It is **not** a display title; it must match Slurm exactly.
- If `sbatch` says *Invalid account* and lists allowed accounts (e.g. `aip-boyuwang`), use **one of those strings verbatim**.
- **Never** instruct anyone to run `sed` that replaces a real account with a **placeholder** like `your-ccdb-group` or `YOURGROUP` — the job will fail. Either keep a real example account in the script or tell them to set the real name from CCDB / `sbatch`’s error list.
- Optional: `export SLURM_ACCOUNT=aip-boyuwang` and `export SBATCH_ACCOUNT=$SLURM_ACCOUNT` in `~/.bashrc`, or override per job: `sbatch --account=aip-boyuwang script.sh`.

## `$SCRATCH` and `$USER` (second common failure mode)

On **Killarney**, `$SCRATCH` is typically **already** your user scratch directory (e.g. `/scratch/ccao87`). It is **not** `/scratch` with a missing `$USER` segment that you must add by hand.

- **Wrong:** `cd $SCRATCH/$USER` when `$SCRATCH=/scratch/ccao87` → becomes `/scratch/ccao87/ccao87` → **No such file or directory**.
- **Right:** `cd $SCRATCH` and clone/checkout the repo there, e.g. **`$SCRATCH/drc-sokoban-ma`** (same pattern as `ts-sandbox` using `$SCRATCH/ts-sandbox`).

Repo job scripts in this project should resolve the working tree like:

```bash
if [ -d "$SCRATCH/drc-sokoban-ma" ]; then
  export PROJECT_ROOT="$SCRATCH/drc-sokoban-ma"
elif [ -d "$HOME/drc-sokoban-ma" ]; then
  export PROJECT_ROOT="$HOME/drc-sokoban-ma"
else
  echo "ERROR: clone repo to \$SCRATCH/drc-sokoban-ma (or \$HOME fallback)"
  exit 1
fi
```

Only use `$SCRATCH/$USER/<repo>` if **your site documentation** says scratch is the parent of per-user dirs (unusual on Killarney for the common case above).

## Accounts and Slurm basics

- **Scheduler:** Slurm only. No compute on login nodes except tiny tasks (~≤10 CPU-minutes, ~≤4 GB RAM). Everything else: `sbatch`, `salloc`, `srun`.
- **Minimum directives:** Always set **`#SBATCH --time=...`**. Add **`--mem`** or **`--mem-per-cpu`** on general-purpose clusters (default can be very small per core). `#SBATCH` lines must come **before** any shell commands in the script.
- **Do not** hammer Slurm with `squeue`/`sq` in tight loops; use mail notifications or reasonable polling.
- **Partitions:** Prefer **L40S + `--gres`** as default; only pin **`gpubase_h100_*`** when the user needs H100. Verify live with `sinfo` on the target system.

## Where to put files (storage hygiene)

- **HOME (`~`):** Small quota; keep **source, job scripts, tiny configs**. Not for large datasets or heavy I/O.
- **SCRATCH:** Large, fast for big sequential I/O; **not backed up**; old files may be **purged** (e.g. 60-day policy—check current docs). Use for **checkpoints during runs**, bulk output, datasets you can re-fetch, and **clone/run the repo** where policy requires (Killarney: **no GPU work from `/home`** — run from `$SCRATCH/...`).
- **PROJECT (`$PROJECT` / `~/projects/...`):** Shared by the allocation group, larger quota, backed up; intended for **relatively static** shared data—frequent churn hurts tape backup. For **your** artifacts, use a **per-user subdirectory**: `$PROJECT/$USER/<app>/` (venv, checkpoints, wandb, copied datasets), **not** the group root as a personal scratch pad.
- **`$SLURM_TMPDIR`:** Per-job local disk on the compute node; great for **many small files** and ephemeral shuffles; **deleted when the job ends**.
- **Python/R packages:** Often **not** full Lmod modules; use Alliance **Python + pip/wheel** docs, or install into **your** space (venv under `$PROJECT/$USER/...` or similar).
- **Modules:** Start job scripts with **`module purge`** then load what you need; prerequisites matter—use **`module spider <name>/<version>`** to see the load chain. Avoid relying on whatever was loaded in the interactive shell.

**Per-user working space (important):** Treat **group project space as shared infrastructure**. Keep **your** working data, venvs, and experiment outputs under **`$PROJECT/$USER/...`** and/or **`$SCRATCH/$USER/...`** (only when that path is valid on your site). Do **not** stash personal experiments or venvs directly under `$PROJECT/` without a `/$USER/` (or agreed team) path—avoids quota fights and policy issues.

## Resolving `$PROJECT` in job scripts (ts-sandbox pattern)

If `$PROJECT` is empty in the batch environment:

```bash
if [ -z "$PROJECT" ] && [ -d "$HOME/projects" ]; then
  FIRST_PROJECT=$(ls -d "$HOME"/projects/def-* "$HOME"/projects/aip-* 2>/dev/null | head -1)
  [ -n "$FIRST_PROJECT" ] && export PROJECT=$(readlink -f "$FIRST_PROJECT")
fi
```

Then set storage, e.g. `STORAGE_ROOT="$PROJECT/$USER/drc-sokoban-ma"` for checkpoints, wandb, venv.

## Killarney — hardware reference (verify with `sinfo`)

This cluster is common for this repo. **Verify live** with `sinfo -o "%P %G %l"` and `scontrol show node | grep -i gres`—names and partitions change.

- **L40S (default tier for scripts in this repo):** Standard compute; request with `#SBATCH --gres=gpu:l40s:1`. Example scripts: `slurm_ci_latent_etth2.sh`, **`slurm_ma_tom.sh`** (default), `slurm_latent_experiment.sh`.
- **H100 (optional, heavy jobs):** Dell XE9680-class nodes, **8× H100 SXM 80GB** per node. Use `#SBATCH --partition=gpubase_h100_b*` + `#SBATCH --gpus-per-node=h100:N` — **not** mixed with L40S `gres`.

- **Code location:** Killarney **must not run GPU work from `/home`**; keep a checkout under **`$SCRATCH/<repo-name>`** (e.g. `$SCRATCH/drc-sokoban-ma`, same idea as `$SCRATCH/ts-sandbox` in the other repo).
- **Account prefix:** Allocations may show as **`aip-...`** on Killarney vs **`def-...`** elsewhere—use the CCDB **Group Name** for `--account`.

## Other clusters (quick reference)

Repo comments (`slurm_pipeline.sh`): **Narval** → e.g. A100; **Fir / Nibi / Rorqual** → e.g. H100-style requests. Always match **`--account`** to an allocation **valid on that cluster** (RAPs are not always portable). Default to **smaller / general GPU** when unsure, not the largest SKU.

## Software modules (typical ML stack)

Example stack used in this repo’s Slurm scripts:

```bash
module purge
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load cudnn/8.9
```

Load **CUDA/cuDNN** versions compatible with your PyTorch build. If a module fails, use `module spider` to resolve prerequisites. **Docker** is not available; **Apptainer** is (`module load apptainer`).

## Patterns from this repository

- **`#SBATCH --account=aip-boyuwang`** — example only; **replace with your real CCDB group name** if different, never with a fake placeholder string.
- **Venv and bulky data:** `$PROJECT/$USER/<app>/venv` (and checkpoints/results under the same tree)—**per-user under project**, not the bare group directory.
- **`slurm_ma_tom.sh`:** defaults to **L40S** (`--gres=gpu:l40s:1`); H100 only via explicit `sbatch` overrides (see script header).
- **`slurm_unet_fullvar.sh` (ts-sandbox):** **L40S** for `--smoke-test`, **H100** + `gpubase_h100_b4` for full runs.
- **`PYTHONUNBUFFERED=1`** and **`python -u`** help with Slurm log latency (buffering).

## Gotchas (see also project `onboard.md` and `.ai/cluster-paths.md`)

- **Empty/broken venv on cluster:** If jobs fail with missing `torch`, recreate or repair **`$PROJECT/$USER/.../venv`** (or delete and let the Slurm script reinstall).
- **Imports:** Run Python as **`python -m package.module`** from the repo root.
- **Slurm output buffering** can make logs look “stuck”; use unbuffered Python or interactive `salloc` to debug.
- **`module purge` in jobs** avoids surprise inherited environments from the submission shell.
- **`sbatch` from scripts:** Do not submit thousands of jobs at once; prefer **arrays** or spacing submissions—Alliance warns this can harm Slurm.
- **Stale script on cluster:** If `squeue` still shows H100/64G after switching to L40S, **`git pull`** on the cluster clone and **resubmit** — old `#SBATCH` lines are baked in at submit time.

## Official docs

Prefer [docs.alliancecan.ca](https://docs.alliancecan.ca/) for authoritative quotas, partition names, and policy updates—this skill is a condensed assistant checklist, not a substitute for current site documentation.
