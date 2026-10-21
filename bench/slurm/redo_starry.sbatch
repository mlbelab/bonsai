#!/bin/bash
# M2 upstream baselines (A22): stage-limited (stopped after redo starry) simulated 1k and 10k, seeds 0-2.
#
# Run from a dedicated worktree pinned to a reviewed sha, never the main checkout (same discipline
# as baselines.sbatch):
#
#   git worktree add ../bonsai-m2-baselines <sha>
#   cd ../bonsai-m2-baselines
#   uv sync --locked
#   mkdir -p bench/results/slurm_logs
#
# No `#SBATCH --array` line: the two sizes are submitted separately, the 10k batch only after the
# 1k wall times are seen (Q3 ruling: revisit the 10k limit if a 1k job exceeds 3h).
#
#   sbatch --array=0-2 bench/slurm/redo_starry.sbatch   # 1k, seeds 0,1,2
#   sbatch --array=3-5 bench/slurm/redo_starry.sbatch   # 10k, seeds 0,1,2
#
#SBATCH --job-name=bonsai-m2-redo-starry
#SBATCH --partition=batch_cpu
#SBATCH --qos=15d
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --no-requeue
#SBATCH --output=bench/results/slurm_logs/%x-%A_%a.out
#SBATCH --error=bench/results/slurm_logs/%x-%A_%a.err

set -euo pipefail

# SLURM runs the job in the directory it was submitted from (no --chdir here, since the dedicated
# worktree's path is chosen at `git worktree add` time, not known in advance): guard against running
# in the wrong tree (e.g. the main checkout, or a stale worktree) by requiring this very script to be
# present relative to the current directory and that directory to be a git working tree.
if [[ ! -f bench/slurm/redo_starry.sbatch ]] || ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "error: not running from the dedicated baseline worktree's repo root" \
       "(bench/slurm/redo_starry.sbatch not found under \$(pwd), or \$(pwd) is not a git checkout)." \
       "Submit from the worktree created per this script's header comment." >&2
  exit 1
fi

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cpu

i="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set: submit with --array=0-2 (1k) or --array=3-5 (10k)}"
sizes=(1k 10k)
size="${sizes[$((i / 3))]}"
seed=$((i % 3))

out="${BENCH_OUT:-bench/results}"
mkdir -p "$out" bench/results/slurm_logs

timeout 47h uv run --locked python -m bench.run \
  --config "bench/configs/upstream_redo_sim_${size}.toml" \
  --set "run.seed=${seed}" \
  --set "run.name=upstream_redo_sim_${size}_s${seed}" \
  --out "$out" \
  > "$out/upstream_redo_sim_${size}_s${seed}.sbatch.log" 2>&1
