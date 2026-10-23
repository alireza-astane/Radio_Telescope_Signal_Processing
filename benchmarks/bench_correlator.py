"""Correlator throughput: X-step (and F-step) time vs number of antennas,
channels and backend.  Writes benchmarks/results/correlator*.csv.

The X-step does ``n_bl * n_samples`` complex multiply-accumulates (8 real
flops each), so its cost grows as N^2.  The real-time factor is the duration
of the data (``n_samples / sample_rate``) divided by the processing time:
> 1 means the backend keeps up with a live stream at that rate.

``--quick`` runs a reduced sweep (used by CI).  CUDA rows are written as
``status=pending`` placeholders when no GPU is available; run
``benchmarks/gpu_colab.ipynb`` on a GPU machine to fill them in.
"""

from __future__ import annotations

import argparse
import csv

import numpy as np

from _common import RESULTS, best_time, cpu_model, git_hash

from radiotel import correlator, signals
from radiotel.correlator import c_fx, torch_fx

SAMPLE_RATE = 2.4e6
FIELDS = ["backend", "n_ant", "nchan", "n_samples", "threads", "t_fstep_s", "t_xstep_s", "gflops_x",
          "rtf_xstep", "rtf_total", "status", "device", "cpu", "git"]
# Skip configurations whose estimated X-step time exceeds this (slow reference backends).
MAX_EST_SECONDS = 20.0
SLOW = {"numpy-loop": 2.0e8, "c-scalar": 1.5e9}  # rough complex MAC/s, for skipping only


def gpu_name() -> str:
    """Recorded in the CSV so GPU rows say which card produced them."""
    return f"cuda: {torch_fx.torch.cuda.get_device_name(0)}"


def fstep_time(volts, nchan, backend):
    if backend.startswith("torch-"):
        dev = backend.split("-", 1)[1]
        vt = torch_fx.to_device(volts, dev)
        return best_time(lambda: torch_fx.channelize(vt, nchan), sync=lambda: torch_fx.synchronize(dev))
    return best_time(lambda: np.fft.fft(volts.reshape(volts.shape[0], -1, nchan), axis=-1, norm="ortho"))


def xstep_time(spectra, backend):
    if backend.startswith("torch-"):
        dev = backend.split("-", 1)[1]
        st = torch_fx.torch.as_tensor(spectra, device=dev)
        with torch_fx.torch.no_grad():
            return best_time(lambda: torch_fx.xstep(st), sync=lambda: torch_fx.synchronize(dev))
    return best_time(lambda: correlator.xstep(spectra, backend), repeats=3)


def row(backend, n_ant, nchan, n_samp, threads, tf, tx, status="ok", device="cpu"):
    n_bl = n_ant * (n_ant - 1) // 2
    secs = n_samp / SAMPLE_RATE
    ok = status == "ok"
    return {
        "backend": backend, "n_ant": n_ant, "nchan": nchan, "n_samples": n_samp, "threads": threads,
        "t_fstep_s": f"{tf:.6g}" if ok else "", "t_xstep_s": f"{tx:.6g}" if ok else "",
        "gflops_x": f"{8 * n_bl * n_samp / tx / 1e9:.4g}" if ok else "",
        "rtf_xstep": f"{secs / tx:.4g}" if ok else "", "rtf_total": f"{secs / (tf + tx):.4g}" if ok else "",
        "status": status, "device": device, "cpu": cpu_model(), "git": git_hash(),
    }


def sweep(n_ants, nchans, n_samp, backends, rng):
    rows = []
    threads = c_fx.max_threads() if c_fx.available() else 1
    for nchan in nchans:
        for n_ant in n_ants:
            volts = signals.complex_noise(rng, (n_ant, n_samp)).astype(np.complex64)
            spectra = correlator.channelize(volts, nchan).astype(np.complex64)
            n_bl = n_ant * (n_ant - 1) // 2
            for b in backends:
                if b in SLOW and n_bl * n_samp / SLOW[b] > MAX_EST_SECONDS:
                    continue
                tf = fstep_time(volts, nchan, b)
                tx = xstep_time(spectra, b)
                r = row(b, n_ant, nchan, n_samp, threads if b == "c-avx2-omp" else 1, tf, tx,
                        device=gpu_name() if b == "torch-cuda" else "cpu")
                rows.append(r)
                print(f"{b:11s} N={n_ant:3d} nchan={nchan:5d}  X {tx * 1e3:9.2f} ms  "
                      f"{r['gflops_x']:>7s} GFLOP/s  RTF {r['rtf_xstep']}")
    return rows


def thread_scaling(n_ant, nchan, n_samp, rng):
    volts = signals.complex_noise(rng, (n_ant, n_samp)).astype(np.complex64)
    spectra = correlator.channelize(volts, nchan).astype(np.complex64)
    rows, maxt = [], c_fx.max_threads()
    for t in sorted({1, 2, 4, 6, 8, maxt} & set(range(1, maxt + 1))):
        c_fx.set_threads(t)
        tx = best_time(lambda: c_fx.xstep(spectra, "avx2_omp"))
        rows.append(row("c-avx2-omp", n_ant, nchan, n_samp, t, 0.0, tx))
        print(f"threads={t:2d}  X {tx * 1e3:8.2f} ms")
    c_fx.set_threads(maxt)
    return rows


def write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--gpu-only", action="store_true", help="only torch-cuda rows (Colab)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    RESULTS.mkdir(parents=True, exist_ok=True)

    if a.quick:
        n_ants, nchans, n_samp = [2, 4, 8], [256], 1 << 16
    else:
        n_ants, nchans, n_samp = [2, 4, 8, 16, 32, 64], [256, 1024, 4096], 1 << 18

    avail = correlator.available_backends()
    backends = ["torch-cuda"] if a.gpu_only else [b for b in correlator.BACKENDS if b in avail]
    rows = sweep(n_ants, nchans, n_samp, backends, rng)

    if "torch-cuda" not in avail and not a.gpu_only:
        for nchan in nchans:
            for n_ant in n_ants:
                rows.append(row("torch-cuda", n_ant, nchan, n_samp, 1, 0, 0,
                                status="pending: no CUDA device on benchmark host", device="cuda"))

    name = a.out or ("correlator_gpu.csv" if a.gpu_only else "correlator_quick.csv" if a.quick else "correlator.csv")
    write(RESULTS / name, rows)
    if not a.quick and not a.gpu_only and "c-avx2-omp" in avail:
        write(RESULTS / "correlator_threads.csv", thread_scaling(32, 1024, n_samp, rng))


if __name__ == "__main__":
    main()
