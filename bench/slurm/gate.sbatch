#!/bin/bash
# M2 gate sweep (A17): one array task is one line of a sweep manifest (bench.sweep).
#   kind=upstream: re-time only the upstream arm into the shared cache (`bench.gate --upstream-only`).
#   kind=jax:      `bench.run` for the jax arm, then `bench.gate` (the jax arm's re-time, plus the upstream
#                  arm's, which is a cache hit once its upstream line has finished).
#   kind=timing:   R2 timing of one 10k seed: `bench.run` for M = 64, 256, 1024 back to back on this one GPU
#                  (false_stop_check=false, each in its own run dir), then the quality gates of M=64 and 1024
#                  (M=256's gate is the R1 line's). `decide` takes R2 timing only from these runs.
# The re-times of a line run serially.
#
# Setup, from a dedicated worktree pinned to a reviewed sha (same discipline as redo_starry.sbatch):
#
#   git worktree add ../bonsai-m2-gate <sha> && cd ../bonsai-m2-gate
#   uv sync --locked --extra cuda12
#   R=/abs/out/dir; G=<upstream_redo_golden run dir>
#   uv run --locked python -m bench.sweep manifest --wave 1 --out $R/w1.tsv --root $R --golden-upstream $G
#       (prints e.g. "27 lines; upstream 0-8, jax 9-26")
#
# Wave 1 (SWEEP_MANIFEST must be absolute; SWEEP_SHA pins the checkout; add %K after the range to throttle):
#
#   exp=SWEEP_MANIFEST=$R/w1.tsv,SWEEP_SHA=$(git rev-parse HEAD)
#   up=$(sbatch --parsable --export=ALL,$exp --array=0-8 bench/slurm/gate.sbatch)
#   sbatch --export=ALL,$exp --array=9-26 --dependency=afterok:$up bench/slurm/gate.sbatch
#
# Then `python -m bench.sweep aggregate --manifest $R/w1.tsv --out w1.md` gives R1's winner; wave 2:
#
#   uv run --locked python -m bench.sweep manifest --wave 2 --mode <winner> --out $R/w2.tsv --root $R --golden-upstream $G
#   sbatch --export=ALL,SWEEP_MANIFEST=$R/w2.tsv,SWEEP_SHA=... --array=0-32 bench/slurm/gate.sbatch
#       (timing lines are the last 3 ids; ids restart at 0 per wave, output paths depend only on case and setting)
#   python -m bench.sweep aggregate --manifest $R/w1.tsv $R/w2.tsv --out measurements.md
#
# `afterok` holds the jax lines until every upstream line succeeded (if one fails, scancel the held job, fix and
# resubmit: finished lines are skipped). Lines sharing an upstream arm would otherwise race on the cache: that is
# safe (atomic rename: duplicate work, never corruption), the upstream-only lines just avoid the duplicates.
# Resubmission is idempotent: a line whose gate.json (or upstream/timing json) exists is skipped; inside a timing
# line a finished gate is skipped. A line whose gate crashed after `bench.run` reruns the jax run (a new
# timestamp dir) but re-uses every finished re-time. Each timing line is one job on one GPU, so its three
# M runs are interleaved on the same device by construction.
# Logs: <gate output dir>/task-<jobid>_<index>.log, next to the outputs, never in the repo. Before the manifest line is
# known (guard failures, a bad manifest) SLURM's own output goes to ~/bonsai-m2-gate-guard-<jobid>_<index>.log
# (stdout and stderr merged; override with `sbatch --output=...`).
#
# Resources: a 10k line is the jax run (cold + warm) plus two float64 re-times of up to ~4.8 h each on an L40S
# (measured/estimated); an A100's fp64 rate is much higher, so ~10 h worst case. A timing line is three runs plus
# two such re-times. qos 1d (max 24 h) with a 20 h limit leaves a margin without needing the 3d qos queue; the 1k,
# golden and real lines are far shorter.
#SBATCH --job-name=bonsai-m2-gate
#SBATCH --partition=batch_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --qos=1d
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --no-requeue
#SBATCH --output=/home/fleckj/bonsai-m2-gate-guard-%A_%a.log

set -euo pipefail

# Same guard as redo_starry.sbatch (the job runs in the submit directory), plus the pinned sha.
if [[ ! -f bench/slurm/gate.sbatch ]] || ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "error: not running from the dedicated gate worktree's repo root (bench/slurm/gate.sbatch not found" \
       "under $(pwd), or not a git checkout). Submit from the worktree created per this script's header." >&2
  exit 1
fi
if [[ -n "${SWEEP_SHA:-}" && "$(git rev-parse HEAD)" != "$SWEEP_SHA" ]]; then
  echo "error: worktree HEAD $(git rev-parse HEAD) is not the pinned SWEEP_SHA $SWEEP_SHA" >&2
  exit 1
fi

manifest="${SWEEP_MANIFEST:?SWEEP_MANIFEST not set: see the header}"
i="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set: submit with --array}"
field() {  # field NAME -> that column of data line $i (0-based); empty fields stay empty
  awk -F'\t' -v n="$1" -v r="$((i + 2))" 'NR==1{for(k=1;k<=NF;k++)if($k==n)c=k} NR==r{print $c}' "$manifest"
}
kind="$(field kind)"; setting="$(field setting)"
upstream="$(field upstream)"; gate_out="$(field gate_out)"; cache="$(field cache)"
[[ -n "$kind" && -n "$gate_out" ]] || { echo "error: no manifest line $i in $manifest" >&2; exit 1; }

mkdir -p "$(dirname "$gate_out")"
exec >> "$(dirname "$gate_out")/task-${SLURM_ARRAY_JOB_ID:-0}_${i}.log" 2>&1
echo "== line $i: $kind $setting ($gate_out) on $(hostname), started $(date -u +%FT%TZ), sha $(git rev-parse HEAD)"
nvidia-smi -L || true

if [[ -s "$gate_out" ]]; then
  echo "skip: $gate_out exists"
  exit 0
fi

n="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="$n" MKL_NUM_THREADS="$n" OPENBLAS_NUM_THREADS="$n"
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
run=(uv run --locked --extra cuda12 python -m)

jax_args() {  # common bench.run arguments of this line
  args=(--config "$(field config)" --out "$(field run_out)")
  [[ -z "$(field data_path)" ]] || args+=(--data-path "$(field data_path)")
  IFS=';' read -ra sets <<< "$(field sets)"
  for s in "${sets[@]}"; do args+=(--set "$s"); done
}

if [[ "$kind" == upstream ]]; then
  "${run[@]}" bench.gate --upstream "$upstream" --upstream-only --retime-cache "$cache" --device 0 --out "$gate_out"
elif [[ "$kind" == jax ]]; then
  jax_args
  run_dir="$("${run[@]}" bench.run "${args[@]}" | tail -n 1)"
  echo "jax run dir: $run_dir"
  "${run[@]}" bench.gate --jax "$run_dir" --upstream "$upstream" --retime-cache "$cache" --device 0 --out "$gate_out"
else  # timing
  jax_args
  IFS=',' read -ra Ms <<< "$(field rescore_top)"
  mode="$(field mode)"
  declare -A rd
  for M in "${Ms[@]}"; do
    rd[$M]="$("${run[@]}" bench.run "${args[@]}" --set "run.name=$(field run_name)_M${M}_m1_timing" \
      --set "jax.rescore_top=$M" | tail -n 1)"
    echo "timing run M=$M: ${rd[$M]}"
  done
  for M in "${Ms[@]}"; do
    [[ "$M" == 256 ]] && continue  # the R1 line gates M=256
    g="$(field gate_base)/${mode}_M${M}_m1/gate.json"
    mkdir -p "$(dirname "$g")"
    [[ -s "$g" ]] || "${run[@]}" bench.gate --jax "${rd[$M]}" --upstream "$upstream" --retime-cache "$cache" \
      --device 0 --out "$g"
  done
  js="{"; sep=""
  for M in "${Ms[@]}"; do js+="$sep\"$M\": \"${rd[$M]}\""; sep=", "; done
  echo "$js}" > "$gate_out.tmp" && mv "$gate_out.tmp" "$gate_out"
fi
echo "== line $i done $(date -u +%FT%TZ)"
