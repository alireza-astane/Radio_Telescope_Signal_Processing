"""Two-signal phase-difference estimators (the original project question).

All estimators return the phase of ``s2`` relative to ``s1`` wrapped to
``(-pi, pi]``, for ``s2(t) = sin(w t + phi)`` and ``s1(t) = sin(w t)``.
They are cleaned-up versions of the estimators compared in
``legacy/methods.ipynb``.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sps


def wrap(phi):
    """Wrap angles to ``(-pi, pi]``."""
    return np.pi - np.mod(np.pi - np.asarray(phi), 2 * np.pi)


def _parabolic_peak(y: np.ndarray, k: int) -> float:
    """Sub-sample peak location from a parabola through ``y[k-1:k+2]``."""
    if k <= 0 or k >= len(y) - 1:
        return float(k)
    a, b, c = y[k - 1], y[k], y[k + 1]
    denom = a - 2 * b + c
    return k + (0.5 * (a - c) / denom if denom != 0 else 0.0)


def estimate_frequency(s: np.ndarray, sample_rate: float) -> float:
    """Dominant frequency (Hz) from the interpolated magnitude-spectrum peak."""
    spec = np.abs(np.fft.rfft(s * np.hanning(len(s))))
    k = int(np.argmax(spec[1:])) + 1
    return _parabolic_peak(np.log(spec + 1e-300), k) * sample_rate / len(s)


def fft_peak(s1, s2, sample_rate=None, freq=None) -> float:
    """Phase of the cross-spectrum at the strongest bin.

    Taking ``angle(X2 X1*)`` in the *same* bin cancels the leakage phase that
    biases subtracting two independently-measured bin phases.
    """
    x1, x2 = np.fft.rfft(s1), np.fft.rfft(s2)
    k = int(np.argmax(np.abs(x1) + np.abs(x2)))
    return float(wrap(np.angle(x2[k] * np.conj(x1[k]))))


def hilbert_unwrap(s1, s2, sample_rate=None, freq=None) -> float:
    """Legacy method: mean of the unwrapped instantaneous-phase difference."""
    p1 = np.unwrap(np.angle(sps.hilbert(s1)))
    p2 = np.unwrap(np.angle(sps.hilbert(s2)))
    return float(wrap(np.mean(p2 - p1)))


def hilbert_cross(s1, s2, sample_rate=None, freq=None) -> float:
    """Phase of the averaged analytic cross-product ``<a2 a1*>`` (no unwrapping)."""
    a1, a2 = sps.hilbert(s1), sps.hilbert(s2)
    return float(np.angle(np.sum(a2 * np.conj(a1))))


def xcorr_lag(s1, s2, sample_rate, freq=None) -> float:
    """Time-domain cross-correlation lag, converted to phase.

    The legacy version took the integer arg-max over all lags, quantising the
    answer to whole samples and allowing jumps of whole periods.  Here the
    search is restricted to +-half a period and refined with a parabola.
    """
    freq = freq or estimate_frequency(s1, sample_rate)
    corr = sps.correlate(s2, s1, mode="full", method="fft")
    lags = sps.correlation_lags(len(s2), len(s1), mode="full")
    half_period = 0.5 * sample_rate / freq
    window = np.abs(lags) <= max(half_period, 1.0)
    idx = np.flatnonzero(window)
    k = idx[np.argmax(corr[idx])]
    lag = _parabolic_peak(corr, k) - k + lags[k]
    return float(wrap(-2 * np.pi * freq * lag / sample_rate))


ESTIMATORS = {
    "fft_peak": fft_peak,
    "hilbert_unwrap": hilbert_unwrap,
    "hilbert_cross": hilbert_cross,
    "xcorr_lag": xcorr_lag,
}


def crlb_phase_difference(n_samples: int, snr_linear: float) -> float:
    """Cramér–Rao lower bound on ``var(phi2 - phi1)`` for two noisy real tones.

    For one real tone of known frequency, ``var(phi) >= 1 / (N * SNR)`` with
    ``SNR = A^2 / (2 sigma^2)``; the difference of two independent estimates
    doubles it.
    """
    return 2.0 / (n_samples * snr_linear)
