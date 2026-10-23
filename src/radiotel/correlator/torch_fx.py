"""PyTorch FX correlator (CPU or CUDA).

Same GEMM formulation as ``numpy_fx.xstep_gemm``; on a GPU the per-channel
``S @ S^H`` products map onto batched complex matrix multiplies.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
except ImportError:  # optional dependency
    torch = None

from ..geometry import baseline_pairs


def available(device: str = "cpu") -> bool:
    if torch is None:
        return False
    return device == "cpu" or (device.startswith("cuda") and torch.cuda.is_available())


def default_device() -> str:
    return "cuda" if available("cuda") else "cpu"


def to_device(volts: np.ndarray, device: str):
    return torch.as_tensor(np.ascontiguousarray(volts, dtype=np.complex64), device=device)


def channelize(volts_t, nchan: int):
    n_ant, n_samp = volts_t.shape
    n_spec = n_samp // nchan
    blocks = volts_t[:, : n_spec * nchan].reshape(n_ant, n_spec, nchan)
    return torch.fft.fft(blocks, dim=-1, norm="ortho")


def xstep(spectra_t):
    n_ant, n_spec, _ = spectra_t.shape
    s = spectra_t.permute(2, 0, 1)  # (nchan, n_ant, n_spec)
    r = torch.matmul(s, s.conj().transpose(1, 2)) / n_spec
    ii, jj = (torch.as_tensor(x, device=spectra_t.device) for x in baseline_pairs(n_ant))
    return r[:, ii, jj].T


def correlate(volts: np.ndarray, nchan: int, device: str | None = None) -> np.ndarray:
    """Voltages -> visibilities; transfers in and out of the device included."""
    device = device or default_device()
    with torch.no_grad():
        vis = xstep(channelize(to_device(volts, device), nchan))
    return vis.cpu().numpy()


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
