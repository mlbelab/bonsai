"""Run metadata for benchmark results (spec §4 "Benchmarks").

Adapted from rose ``bench/metadata.py`` (mlbelab internal; behaviour reproduced, trimmed).
"""

import hashlib
import importlib.metadata
import os
import platform
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")  # timing validity: record them
PACKAGES = ("bonsai", "jax", "jaxlib", "equinox", "optimistix", "numpy", "scipy", "anndata", "scikit-learn")


def _run(*cmd: str) -> str | None:
    try:
        return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def _version(pkg: str) -> str | None:
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect(config_path: Path, seed: int) -> dict:
    """Environment and provenance of one benchmark run; unknown fields are None, never an error."""
    # --untracked-files=no: bench/results/ is committed only after Task 5, so every baseline run
    # before that would otherwise see its own (untracked) output files and report a dirty tree.
    status = _run("git", "status", "--porcelain", "--untracked-files=no")
    gpus = _run("nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader")
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git_sha": _run("git", "rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "cpu_count": os.cpu_count(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "threads": {v: os.environ.get(v) for v in THREAD_VARS},
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "gpus": gpus.splitlines() if gpus else [],
        "versions": {p: _version(p) for p in PACKAGES},
        "seed": seed,
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        "config_text": Path(config_path).read_text(),
    }
