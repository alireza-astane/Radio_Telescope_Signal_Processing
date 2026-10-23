"""Antenna-based gain calibration and closure phases.

A receiver chain multiplies antenna ``k``'s voltage by a complex gain
``g_k`` (cable delay, LNA, oscillator phase), so every baseline is corrupted
as ``V_ij^obs = g_i conj(g_j) V_ij``.  With ``N`` antennas there are only
``N`` unknowns but ``N(N-1)/2`` baselines, so gains can be solved from a
calibrator observation; closure phases are immune to them altogether.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from .geometry import baseline_pairs


def to_matrix(vis_bl: np.ndarray, n_ant: int) -> np.ndarray:
    """Baseline-ordered ``(..., n_bl)`` visibilities -> Hermitian ``(..., n_ant, n_ant)``."""
    ii, jj = baseline_pairs(n_ant)
    r = np.zeros(vis_bl.shape[:-1] + (n_ant, n_ant), dtype=complex)
    r[..., ii, jj] = vis_bl
    r[..., jj, ii] = np.conj(vis_bl)
    return r


def stefcal(vis_obs: np.ndarray, vis_model: np.ndarray, n_ant: int, n_iter: int = 200,
            tol: float = 1e-10, ref_ant: int = 0) -> np.ndarray:
    """Solve per-antenna complex gains with StEFCal (Salvini & Wijnholds 2014).

    ``vis_obs`` / ``vis_model``: ``(..., n_bl)`` with baselines on the last
    axis; all other axes (time, channel) are treated as extra samples of the
    same gains.  Returns ``g`` with ``arg g[ref_ant] = 0``.
    """
    n_bl = len(baseline_pairs(n_ant)[0])
    r = to_matrix(np.asarray(vis_obs).reshape(-1, n_bl), n_ant)
    m = to_matrix(np.broadcast_to(vis_model, np.shape(vis_obs)).reshape(-1, n_bl), n_ant)
    g = np.ones(n_ant, dtype=complex)
    for it in range(n_iter):
        z = m * np.conj(g)[None, None, :]  # z[s, p, q] = M_pq conj(g_q)
        g_new = np.sum(r * np.conj(z), axis=(0, 2)) / np.sum(np.abs(z) ** 2, axis=(0, 2))
        if it % 2 == 1:  # damping step that makes StEFCal converge robustly
            g_new = 0.5 * (g_new + g)
        done = np.max(np.abs(g_new - g)) / np.max(np.abs(g_new)) < tol
        g = g_new
        if done:
            break
    return g * np.exp(-1j * np.angle(g[ref_ant]))


def apply_gains(vis_bl: np.ndarray, gains: np.ndarray, invert: bool = True) -> np.ndarray:
    """Remove (``invert=True``) or apply antenna gains to baseline-ordered visibilities."""
    ii, jj = baseline_pairs(len(gains))
    gg = gains[ii] * np.conj(gains[jj])
    return vis_bl / gg if invert else vis_bl * gg


def closure_phases(vis_bl: np.ndarray, n_ant: int) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """Closure phases ``arg V_ij + arg V_jk - arg V_ik`` for every triangle ``i<j<k``.

    Returns ``(phases (..., n_tri), triangles)``.
    """
    ii, jj = baseline_pairs(n_ant)
    index = {(i, j): b for b, (i, j) in enumerate(zip(ii, jj))}
    tris = list(combinations(range(n_ant), 3))
    out = np.stack(
        [
            np.angle(vis_bl[..., index[i, j]] * vis_bl[..., index[j, k]] * np.conj(vis_bl[..., index[i, k]]))
            for i, j, k in tris
        ],
        axis=-1,
    )
    return out, tris
