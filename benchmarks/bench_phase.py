"""Two-signal phase estimators: accuracy vs SNR (against the Cramér–Rao bound),
robustness to tone frequency, and runtime vs signal length.

This is the reproducible version of the comparison in legacy/methods.ipynb.
Writes benchmarks/results/phase_*.csv.
"""

from __future__ import annotations

import argparse
import csv

import numpy as np

from _common import RESULTS, best_time, cpu_model, git_hash

from radiotel import phase, signals

FS = 1024.0
TRUE_PHI = 1.0


def mc_errors(est, freq, n, snr_db, trials, rng):
    errs = np.empty(trials)
    for k in range(trials):
        _, s1, s2 = signals.two_tone_signals(freq, FS, n, TRUE_PHI, snr_db, rng)
        errs[k] = phase.wrap(est(s1, s2, FS, freq) - TRUE_PHI)
    return errs


def accuracy_vs_snr(trials, rng):
    rows, n, freq = [], 1024, 97.3
    for snr_db in np.arange(-15, 31, 3):
        bound = phase.crlb_phase_difference(n, 10 ** (snr_db / 10))
        for name, est in phase.ESTIMATORS.items():
            e = mc_errors(est, freq, n, snr_db, trials, rng)
            rows.append({"estimator": name, "snr_db": snr_db, "n_samples": n, "freq_hz": freq,
                         "rmse_rad": np.sqrt(np.mean(e**2)), "bias_rad": np.mean(e),
                         "crlb_rmse_rad": np.sqrt(bound), "trials": trials})
        print(f"SNR {snr_db:4.0f} dB done")
    return rows


def accuracy_vs_freq(trials, rng):
    """Noise-free sweep across the band: exposes edge effects near DC and Nyquist."""
    rows, n = [], 1024
    for freq in np.concatenate([np.linspace(2.3, 500.0, 40), [505.0, 509.0]]):
        for name, est in phase.ESTIMATORS.items():
            e = mc_errors(est, freq, n, 30.0, trials, rng)
            rows.append({"estimator": name, "freq_hz": round(float(freq), 3), "snr_db": 30.0,
                         "n_samples": n, "rmse_rad": np.sqrt(np.mean(e**2)), "trials": trials})
    return rows


def runtime_vs_length(rng):
    rows = []
    for log2n in range(8, 21, 2):
        n = 1 << log2n
        _, s1, s2 = signals.two_tone_signals(97.3, FS, n, TRUE_PHI, 20.0, rng)
        for name, est in phase.ESTIMATORS.items():
            t = best_time(lambda: est(s1, s2, FS, 97.3), repeats=5)
            rows.append({"estimator": name, "n_samples": n, "time_s": t, "cpu": cpu_model(), "git": git_hash()})
        print(f"N = 2^{log2n} done")
    return rows


def write(name, rows):
    path = RESULTS / name
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=int, default=400)
    a = ap.parse_args()
    rng = np.random.default_rng(42)
    RESULTS.mkdir(parents=True, exist_ok=True)
    write("phase_vs_snr.csv", accuracy_vs_snr(a.trials, rng))
    write("phase_vs_freq.csv", accuracy_vs_freq(max(a.trials // 10, 10), rng))
    write("phase_runtime.csv", runtime_vs_length(rng))


if __name__ == "__main__":
    main()
