"""Reference FX correlator in NumPy.

F-step: split each antenna's stream into ``n_spec`` blocks of ``nchan``
samples and FFT them (unitary normalisation, so a source of flux S gives
|V| = S) -> ``spectra[n_ant, n_spec, nchan]``.

X-step: for every baseline (i < j) and channel, average
``spectra_i * conj(spectra_j)`` over the blocks.  The X-step costs
``O(n_ant^2 * n_samples)`` and dominates for large arrays.
"""

from __future__ import annotations

import numpy as np

from ..geometry import baseline_pairs


def channelize(volts: np.ndarray, nchan: int) -> np.ndarray:
    """F-step: ``(n_ant, n_samp)`` -> ``(n_ant, n_spec, nchan)`` spectra."""
    n_ant, n_samp = volts.shape
    n_spec = n_samp // nchan
    blocks = volts[:, : n_spec * nchan].reshape(n_ant, n_spec, nchan)
    return np.fft.fft(blocks, axis=-1, norm="ortho")


def xstep_loop(spectra: np.ndarray) -> np.ndarray:
    """X-step, one baseline at a time (the direct translation of the maths)."""
    n_ant, n_spec, nchan = spectra.shape
    ii, jj = baseline_pairs(n_ant)
    vis = np.empty((len(ii), nchan), dtype=np.result_type(spectra.dtype, np.complex64))
    for b, (i, j) in enumerate(zip(ii, jj)):
        vis[b] = np.mean(spectra[i] * np.conj(spectra[j]), axis=0)
    return vis


def xstep_gemm(spectra: np.ndarray) -> np.ndarray:
    """X-step as one batched matrix product per channel.

    ``R[c] = S[c] @ S[c]^H`` with ``S[c]`` of shape ``(n_ant, n_spec)`` gives
    every antenna pair at once; this is the formulation GPU tensor-core
    correlators use.  Only the upper triangle is returned.
    """
    n_ant, n_spec, _ = spectra.shape
    s = np.ascontiguousarray(spectra.transpose(2, 0, 1))  # (nchan, n_ant, n_spec)
    r = s @ np.conj(s.transpose(0, 2, 1))  # (nchan, n_ant, n_ant)
    ii, jj = baseline_pairs(n_ant)
    return (r[:, ii, jj] / n_spec).T


def correlate(volts: np.ndarray, nchan: int, method: str = "gemm") -> np.ndarray:
    """Full FX correlation: voltages ``(n_ant, n_samp)`` -> visibilities ``(n_bl, nchan)``."""
    spectra = channelize(volts, nchan)
    return xstep_gemm(spectra) if method == "gemm" else xstep_loop(spectra)
