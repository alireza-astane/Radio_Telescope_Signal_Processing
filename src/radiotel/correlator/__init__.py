"""FX correlator with interchangeable X-step backends.

>>> vis = correlate(volts, nchan=64, backend="c-avx2-omp")

Backends
--------
``numpy-loop``   one baseline at a time (reference maths)
``numpy-gemm``   per-channel ``S @ S^H`` via BLAS
``torch-cpu``    same GEMM formulation in PyTorch
``torch-cuda``   same, on a CUDA GPU (if available)
``c-scalar``     hand-written C, vectorisation disabled
``c-avx2``       C with AVX2+FMA intrinsics
``c-avx2-omp``   C with AVX2+FMA, OpenMP across baselines

The F-step (channelisation FFT) uses NumPy for the NumPy/C backends and
``torch.fft`` for the Torch backends.
"""

from __future__ import annotations

import numpy as np

from . import c_fx, numpy_fx, torch_fx
from .numpy_fx import channelize

BACKENDS = (
    "numpy-loop",
    "numpy-gemm",
    "torch-cpu",
    "torch-cuda",
    "c-scalar",
    "c-avx2",
    "c-avx2-omp",
)


def available_backends() -> list[str]:
    out = ["numpy-loop", "numpy-gemm"]
    if torch_fx.available("cpu"):
        out.append("torch-cpu")
    if torch_fx.available("cuda"):
        out.append("torch-cuda")
    if c_fx.available():
        out.append("c-scalar")
        if c_fx.has_avx2():
            out += ["c-avx2", "c-avx2-omp"]
    return out


def xstep(spectra: np.ndarray, backend: str) -> np.ndarray:
    """Run only the X-step on precomputed ``(n_ant, n_spec, nchan)`` spectra."""
    if backend == "numpy-loop":
        return numpy_fx.xstep_loop(spectra)
    if backend == "numpy-gemm":
        return numpy_fx.xstep_gemm(spectra)
    if backend.startswith("torch-"):
        device = backend.split("-", 1)[1]
        with torch_fx.torch.no_grad():
            s = torch_fx.torch.as_tensor(np.ascontiguousarray(spectra, np.complex64), device=device)
            return torch_fx.xstep(s).cpu().numpy()
    if backend.startswith("c-"):
        return c_fx.xstep(spectra, backend[2:].replace("-", "_"))
    raise ValueError(f"unknown backend {backend!r}; choose from {BACKENDS}")


def correlate(volts: np.ndarray, nchan: int, backend: str = "numpy-gemm") -> np.ndarray:
    """Voltages ``(n_ant, n_samp)`` -> visibilities ``(n_bl, nchan)``."""
    if backend.startswith("torch-"):
        return torch_fx.correlate(volts, nchan, device=backend.split("-", 1)[1])
    spectra = channelize(np.asarray(volts, dtype=np.complex64), nchan)
    return xstep(spectra, backend)


__all__ = ["BACKENDS", "available_backends", "channelize", "correlate", "xstep"]
