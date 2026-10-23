"""ctypes bindings for ``libfxcorr.so`` (built from ``csrc/`` with ``make lib``)."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

from ..geometry import baseline_pairs

_LIB_PATH = Path(os.environ.get("RADIOTEL_LIB", Path(__file__).with_name("libfxcorr.so")))
_lib = None

_FLOAT_P = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
KERNELS = ("scalar", "avx2", "avx2_omp")


def load():
    """Load the shared library once; raises ``OSError`` if it has not been built."""
    global _lib
    if _lib is None:
        lib = ctypes.CDLL(str(_LIB_PATH))
        for name in KERNELS:
            fn = getattr(lib, f"fx_xstep_{name}")
            fn.argtypes = [_FLOAT_P, ctypes.c_int, ctypes.c_int, ctypes.c_int, _FLOAT_P]
            fn.restype = None
        lib.fx_cpu_has_avx2.restype = ctypes.c_int
        lib.fx_max_threads.restype = ctypes.c_int
        lib.fx_set_threads.argtypes = [ctypes.c_int]
        lib.fft_forward_batch.argtypes = [_FLOAT_P, ctypes.c_int, ctypes.c_int]
        lib.fft_forward_batch.restype = ctypes.c_int
        _lib = lib
    return _lib


def available() -> bool:
    try:
        load()
        return True
    except OSError:
        return False


def has_avx2() -> bool:
    return bool(load().fx_cpu_has_avx2())


def set_threads(n: int) -> None:
    load().fx_set_threads(int(n))


def max_threads() -> int:
    return int(load().fx_max_threads())


def xstep(spectra: np.ndarray, kernel: str = "avx2_omp") -> np.ndarray:
    """X-step on ``(n_ant, n_spec, nchan)`` spectra -> ``(n_bl, nchan)`` complex64."""
    if kernel not in KERNELS:
        raise ValueError(f"unknown kernel {kernel!r}; choose from {KERNELS}")
    lib = load()
    if kernel != "scalar" and not lib.fx_cpu_has_avx2():
        raise RuntimeError("CPU does not support AVX2+FMA")
    spectra = np.ascontiguousarray(spectra, dtype=np.complex64)
    n_ant, n_spec, nchan = spectra.shape
    n_bl = len(baseline_pairs(n_ant)[0])
    vis = np.empty((n_bl, nchan), dtype=np.complex64)
    getattr(lib, f"fx_xstep_{kernel}")(
        spectra.view(np.float32).reshape(-1), n_ant, n_spec, nchan, vis.view(np.float32).reshape(-1)
    )
    return vis


def fft(blocks: np.ndarray) -> np.ndarray:
    """Forward FFT along the last axis using the C radix-2 FFT (power-of-two lengths)."""
    data = np.array(blocks, dtype=np.complex64, order="C", copy=True)
    n = data.shape[-1]
    if load().fft_forward_batch(data.view(np.float32).reshape(-1), n, data.size // n) != 0:
        raise ValueError("FFT length must be a power of two")
    return data
