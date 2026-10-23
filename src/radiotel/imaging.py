"""Aperture-synthesis imaging: gridding, dirty image/beam and Högbom CLEAN.

Image axes: rows are ``m`` (north up), columns are ``l``; the phase centre is
pixel ``(npix // 2, npix // 2)``.

Gridding is nearest-neighbour.  Rounding ``u`` to the nearest cell causes a
phase error that grows with distance from the phase centre, so the grid is
oversampled by ``pad`` (finer u-v cells, larger FFT) and the image cropped.
The w-term is ignored: for a source at angle ``theta`` the residual phase is
about ``pi * theta^2 * w``, i.e. < 0.35 rad for theta < 0.5 deg on the
~2 km-wavelength baselines here.  Wide-field imaging would need
w-projection or w-stacking.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


def grid(uvw_lambda: np.ndarray, vis: np.ndarray, npix: int, cell_rad: float,
         weighting: str = "natural") -> tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbour gridding of visibilities and their Hermitian conjugates.

    Returns ``(gridded_vis, weights)``, both ``(npix, npix)``.
    """
    du = 1.0 / (npix * cell_rad)
    u = np.concatenate([uvw_lambda[:, 0], -uvw_lambda[:, 0]])
    v = np.concatenate([uvw_lambda[:, 1], -uvw_lambda[:, 1]])
    vals = np.concatenate([vis, np.conj(vis)])

    iu = np.rint(u / du).astype(int) + npix // 2
    iv = np.rint(v / du).astype(int) + npix // 2
    ok = (iu >= 0) & (iu < npix) & (iv >= 0) & (iv < npix)
    iu, iv, vals = iu[ok], iv[ok], vals[ok]

    g = np.zeros((npix, npix), dtype=complex)
    w = np.zeros((npix, npix))
    np.add.at(g, (iv, iu), vals)
    np.add.at(w, (iv, iu), 1.0)
    if weighting == "uniform":
        nz = w > 0
        g[nz] /= w[nz]
        w[nz] = 1.0
    elif weighting != "natural":
        raise ValueError("weighting must be 'natural' or 'uniform'")
    return g, w


def _grid_to_image(g: np.ndarray, wsum: float) -> np.ndarray:
    img = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(g)))
    return np.real(img) * g.size / wsum


def _crop(img: np.ndarray, size: int) -> np.ndarray:
    lo = img.shape[0] // 2 - size // 2
    return img[lo : lo + size, lo : lo + size]


def dirty_image(uvw_lambda, vis, npix, cell_rad, weighting="natural", pad=4) -> np.ndarray:
    """Dirty image ``(npix, npix)`` normalised so that the PSF peak is 1."""
    g, w = grid(uvw_lambda, vis, npix * pad, cell_rad, weighting)
    return _crop(_grid_to_image(g, w.sum()), npix)


def dirty_beam(uvw_lambda, npix, cell_rad, weighting="natural", pad=4) -> np.ndarray:
    """Point-spread function (peak 1 at the centre) matching :func:`dirty_image`.

    With ``pad >= 2`` the beam is returned at ``2 * npix`` so CLEAN can shift
    it anywhere in the image; with ``pad == 1`` it is ``npix`` and periodic.
    """
    g, w = grid(uvw_lambda, np.ones(len(uvw_lambda), complex), npix * pad, cell_rad, weighting)
    return _crop(_grid_to_image(g, w.sum()), npix * min(pad, 2))


def beam_fwhm_pixels(psf: np.ndarray) -> float:
    """Equivalent circular FWHM (pixels) from the main-lobe area above half power."""
    c = psf.shape[0] // 2
    r = 8
    while True:
        patch = psf[c - r : c + r + 1, c - r : c + r + 1]
        edge = np.concatenate([patch[0], patch[-1], patch[:, 0], patch[:, -1]])
        if np.all(edge < 0.5) or r >= c:
            break
        r *= 2
    labels, _ = ndimage.label(patch >= 0.5)
    area = np.sum(labels == labels[r, r])
    return 2.0 * np.sqrt(area / np.pi)


def peak_sidelobe(psf: np.ndarray, fwhm_px: float | None = None) -> float:
    """Largest |PSF| outside 1.5 FWHM of the centre."""
    fwhm_px = fwhm_px or beam_fwhm_pixels(psf)
    n = psf.shape[0]
    y, x = np.mgrid[:n, :n] - n // 2
    return float(np.max(np.abs(psf[np.hypot(x, y) > 1.5 * fwhm_px])))


@dataclass
class CleanResult:
    model: np.ndarray  # CLEAN components (flux per pixel)
    residual: np.ndarray
    restored: np.ndarray
    n_iter: int


def hogbom_clean(dirty: np.ndarray, psf: np.ndarray, gain: float = 0.1, n_iter: int = 2000,
                 threshold: float = 0.0, fwhm_px: float | None = None) -> CleanResult:
    """Högbom CLEAN.

    ``psf`` comes from :func:`dirty_beam` with the same ``npix``, ``cell_rad``
    and ``pad`` as the dirty image: either twice the image size (shifted by
    slicing) or, for ``pad == 1``, the same size (exactly periodic, so shifted
    with wrap-around).
    """
    n = dirty.shape[0]
    if psf.shape[0] == 2 * n:
        def shifted(iy, ix):  # PSF centre (n, n) moved onto (iy, ix)
            return psf[n - iy : 2 * n - iy, n - ix : 2 * n - ix]
    elif psf.shape == dirty.shape:
        def shifted(iy, ix):
            return np.roll(psf, (iy - n // 2, ix - n // 2), axis=(0, 1))
    else:
        raise ValueError("psf must be the image size or twice it")

    residual = dirty.copy()
    model = np.zeros_like(dirty)
    it = 0
    for it in range(1, n_iter + 1):
        iy, ix = np.unravel_index(np.argmax(np.abs(residual)), residual.shape)
        peak = residual[iy, ix]
        if abs(peak) <= threshold:
            it -= 1
            break
        model[iy, ix] += gain * peak
        residual -= gain * peak * shifted(iy, ix)

    restored = restore(model, fwhm_px or beam_fwhm_pixels(psf)) + residual
    return CleanResult(model, residual, restored, it)


def restore(model: np.ndarray, fwhm_px: float) -> np.ndarray:
    """Convolve CLEAN components with a circular Gaussian of peak 1 (the "clean beam")."""
    n = model.shape[0]
    sigma = fwhm_px / (2 * np.sqrt(2 * np.log(2)))
    y, x = np.mgrid[:n, :n] - n // 2
    kernel = np.fft.ifftshift(np.exp(-(x**2 + y**2) / (2 * sigma**2)))
    return np.real(np.fft.ifft2(np.fft.fft2(model) * np.fft.fft2(kernel)))


def lm_to_pixel(l: float, m: float, npix: int, cell_rad: float) -> tuple[int, int]:
    """(row, col) of direction ``(l, m)``."""
    return int(round(m / cell_rad)) + npix // 2, int(round(l / cell_rad)) + npix // 2


def image_extent(npix: int, cell_rad: float) -> list[float]:
    """``imshow`` extent in direction cosines (origin='lower')."""
    lo = (-npix // 2 - 0.5) * cell_rad
    hi = (npix - npix // 2 - 0.5) * cell_rad
    return [lo, hi, lo, hi]
