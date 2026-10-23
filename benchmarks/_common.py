"""Shared helpers for benchmark scripts: import path, machine metadata, timing."""

from __future__ import annotations

import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"
FIGS = ROOT / "figs"
sys.path.insert(0, str(ROOT / "src"))


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "describe", "--always", "--dirty"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def best_time(fn, repeats: int = 3, sync=None) -> float:
    """Minimum wall time of ``repeats`` calls after one warm-up call."""
    fn()
    if sync:
        sync()
    best = float("inf")
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        if sync:
            sync()
        best = min(best, time.perf_counter() - t)
    return best
