"""Helpers for the C streaming tools: write rx_sim input files, read corrd dumps,
and check corrd's output against the offline NumPy correlator."""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from radiotel import geometry, signals, sky as sky_mod  # noqa: E402
from radiotel.correlator import correlate  # noqa: E402

FILE_MAGIC = 0x31445452
DUMP_MAGIC = 0x31565452
DUMP_HEADER = struct.Struct("<IIIIQ")


def write_input(path, iq: np.ndarray) -> None:
    """``iq``: int8 array ``(n_ant, 2 * n_samples)`` of interleaved I/Q."""
    n_ant, n2 = iq.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<IIQ", FILE_MAGIC, n_ant, n2 // 2))
        f.write(np.ascontiguousarray(iq, dtype=np.int8).tobytes())


def read_input(path) -> np.ndarray:
    raw = Path(path).read_bytes()
    magic, n_ant, n_samp = struct.unpack_from("<IIQ", raw)
    assert magic == FILE_MAGIC
    return np.frombuffer(raw, dtype=np.int8, offset=16).reshape(n_ant, 2 * n_samp)


def read_dumps(path):
    """Return ``(vis (n_dump, n_bl, nchan), first_seqs, n_frames)``."""
    raw = Path(path).read_bytes()
    out, seqs, nfr, off = [], [], [], 0
    while off < len(raw):
        magic, n_bl, nchan, n_frames, first = DUMP_HEADER.unpack_from(raw, off)
        assert magic == DUMP_MAGIC
        off += DUMP_HEADER.size
        count = n_bl * nchan
        out.append(np.frombuffer(raw, dtype=np.complex64, count=count, offset=off).reshape(n_bl, nchan))
        off += 8 * count
        seqs.append(first)
        nfr.append(n_frames)
    return np.array(out), np.array(seqs), np.array(nfr)


def make_input(path, n_ant: int, n_samples: int, seed: int = 0) -> None:
    """Simulated campus observation (demo sky) quantised to 8 bits."""
    rng = np.random.default_rng(seed)
    obs = sky_mod.Observation()
    pos = geometry.standard_layouts()["square"] if n_ant == 4 else None
    if pos is None:
        from radiotel.layout import random_layout

        pos = random_layout(n_ant, geometry.CAMPUS_POLYGON_ENU, 15.0, rng)
    volts = signals.simulate_voltages(pos, sky_mod.demo_sky(), obs, 0.0, n_samples=n_samples,
                                      noise_power=1.0, rng=rng)
    write_input(path, signals.quantize_cint8(volts))


def verify(input_path, dump_path, frame: int, nchan: int, integrate: int) -> float:
    """Max relative difference between corrd dumps and the NumPy correlator."""
    volts = signals.dequantize_cint8(read_input(input_path))
    n_samp = volts.shape[1]
    vis, seqs, nfr = read_dumps(dump_path)
    worst = 0.0
    for v, s0, k in zip(vis, seqs, nfr):
        idx = (np.arange(s0 * frame, (s0 + k) * frame)) % n_samp
        ref = correlate(volts[:, idx], nchan, "numpy-gemm")
        worst = max(worst, float(np.max(np.abs(v - ref)) / np.max(np.abs(ref))))
    return worst


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make-input")
    m.add_argument("path")
    m.add_argument("--n-ant", type=int, default=4)
    m.add_argument("--samples", type=int, default=1 << 18)
    m.add_argument("--seed", type=int, default=0)
    v = sub.add_parser("verify")
    v.add_argument("input")
    v.add_argument("dumps")
    v.add_argument("--frame", type=int, default=4096)
    v.add_argument("--nchan", type=int, default=64)
    v.add_argument("--integrate", type=int, default=16)
    a = ap.parse_args()
    if a.cmd == "make-input":
        make_input(a.path, a.n_ant, a.samples, a.seed)
    else:
        err = verify(a.input, a.dumps, a.frame, a.nchan, a.integrate)
        print(f"max relative difference corrd vs numpy: {err:.2e}")
        sys.exit(0 if err < 1e-4 else 1)
