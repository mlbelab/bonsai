#!/bin/bash
# M0 upstream baselines: 1k, 3k, 10k simulated, one after another on one exclusive node (B-M2 + A-5).
#
# Run from a dedicated worktree pinned to a reviewed sha, never the main checkout (a baseline run
# must not see uncommitted or in-flight changes):
#
#   git worktree add ../bonsai-baselines <sha>
#   cd ../bonsai-baselines
#   uv sync --locked
#   mkdir -p bench/results/slurm_logs   # SLURM needs --output/--error's directory to exist already
#   sbatch [-p <partition>] bench/slurm/baselines.sbatch
#
# sbatch directives cannot expand $BENCH_OUT, so SLURM's own stdout/stderr go next to the results
# under bench/results/slurm_logs/ (never the repo root); override with `sbatch --output=... --error=...`
# if BENCH_OUT is overridden at submit time to somewhere else.
#SBATCH --job-name=bonsai-baselines
#SBATCH --partition=batch_cpu
#SBATCH --qos=15d
#SBATCH --time=9-01:00:00
#SBATCH --no-requeue
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --output=bench/results/slurm_logs/%x-%j.out
#SBATCH --error=bench/results/slurm_logs/%x-%j.err

set -euo pipefail

# SLURM runs the job in the directory it was submitted from (no --chdir here, since the dedicated
# worktree's path is chosen at `git worktree add` time, not known in advance): guard against running
# in the wrong tree (e.g. the main checkout, or a stale worktree) by requiring this very script to be
# present relative to the current directory and that directory to be a git working tree.
if [[ ! -f bench/slurm/baselines.sbatch ]] || ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "error: not running from the dedicated baseline worktree's repo root" \
       "(bench/slurm/baselines.sbatch not found under \$(pwd), or \$(pwd) is not a git checkout)." \
       "Submit from the worktree created per this script's header comment." >&2
  exit 1
fi

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cpu

out="${BENCH_OUT:-bench/results}"
mkdir -p "$out" bench/results/slurm_logs

for n in 1k 3k 10k; do
  timeout 72h uv run --locked python -m bench.run \
    --config "bench/configs/upstream_sim_${n}.toml" \
    --out "$out" \
    > "$out/upstream_sim_${n}.sbatch.log" 2>&1
done
