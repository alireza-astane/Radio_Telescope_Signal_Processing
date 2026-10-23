"""Synthetic receiver voltage streams.

Every source is modelled as stationary complex Gaussian noise (as real
astronomical sources are).  Antenna ``a`` receives each source delayed by its
geometric delay relative to the delay-tracked phase centre::

    tau_a = -(u_a l + v_a m + w_a (n - 1)) / c

applied exactly as a phase ramp ``exp(-2 pi i (f0 + f) tau_a)`` in the
frequency domain, so fractional-sample delays are handled correctly.
Independent receiver noise and optional complex gain errors are added per
antenna.
"""

from __future__ import annotations

import numpy as np

from .geometry import C_LIGHT, uvw
from .sky import Observation, Sky


def complex_noise(rng: np.random.Generator, shape, power: float = 1.0) -> np.ndarray:
    """Circular complex Gaussian noise with ``E|x|^2 = power``."""
    scale = np.sqrt(power / 2.0)
    return scale * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))


def antenna_delays(positions, sky: Sky, obs: Observation, hour_angle: float) -> np.ndarray:
    """Geometric delays (s), shape ``(n_src, n_ant)``, after delay tracking."""
    ant_uvw = uvw(positions, hour_angle, obs.dec, obs.lat)  # (n_ant, 3) metres
    lmn = np.array([[s.l, s.m, s.n - 1.0] for s in sky])  # (n_src, 3)
    return -(lmn @ ant_uvw.T) / C_LIGHT


def simulate_voltages(
    positions,
    sky: Sky,
    obs: Observation,
    hour_angle: float,
    n_samples: int | None = None,
    noise_power: float = 1.0,
    gains: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    dtype=np.complex64,
) -> np.ndarray:
    """Simulate one snapshot of complex-baseband voltages, shape ``(n_ant, n_samples)``."""
    rng = rng or np.random.default_rng()
    n_samples = n_samples or obs.samples_per_snapshot
    n_ant = len(positions)

    freqs = obs.f0 + np.fft.fftfreq(n_samples, 1.0 / obs.sample_rate)
    delays = antenna_delays(positions, sky, obs, hour_angle)

    spectra = np.zeros((n_ant, n_samples), dtype=complex)
    for src, tau in zip(sky, delays):
        s = np.fft.fft(complex_noise(rng, n_samples, src.flux))
        spectra += s[None, :] * np.exp(-2j * np.pi * freqs[None, :] * tau[:, None])

    volts = np.fft.ifft(spectra, axis=1)
    volts += complex_noise(rng, volts.shape, noise_power)
    if gains is not None:
        volts *= np.asarray(gains)[:, None]
    return volts.astype(dtype)


def random_gains(n_ant: int, phase_rms_rad: float, amp_rms: float = 0.0, rng=None) -> np.ndarray:
    """Complex per-antenna gain errors (e.g. unequal cable lengths, LNA drifts)."""
    rng = rng or np.random.default_rng()
    amp = 1.0 + amp_rms * rng.standard_normal(n_ant)
    phase = phase_rms_rad * rng.standard_normal(n_ant)
    return amp * np.exp(1j * phase)


def quantize_cint8(volts: np.ndarray, rms_target: float = 24.0) -> np.ndarray:
    """Quantise to interleaved signed 8-bit I/Q, like an SDR front end.

    Returns an ``int8`` array of shape ``(..., 2 * n_samples)``.
    """
    rms = np.sqrt(np.mean(np.abs(volts) ** 2)) or 1.0
    scaled = volts * (rms_target / rms)
    iq = np.empty(volts.shape[:-1] + (2 * volts.shape[-1],), dtype=np.int8)
    iq[..., 0::2] = np.clip(np.round(scaled.real), -127, 127)
    iq[..., 1::2] = np.clip(np.round(scaled.imag), -127, 127)
    return iq


def dequantize_cint8(iq: np.ndarray) -> np.ndarray:
    iq = iq.astype(np.float32)
    return (iq[..., 0::2] + 1j * iq[..., 1::2]).astype(np.complex64)


def two_tone_signals(freq, sample_rate, n_samples, phase_diff, snr_db=np.inf, rng=None):
    """Two real sinusoids offset by ``phase_diff`` plus white noise (legacy test case).

    ``snr_db`` is the per-channel signal-to-noise power ratio ``A^2 / (2 sigma^2)``.
    """
    rng = rng or np.random.default_rng()
    t = np.arange(n_samples) / sample_rate
    s1 = np.sin(2 * np.pi * freq * t)
    s2 = np.sin(2 * np.pi * freq * t + phase_diff)
    if np.isfinite(snr_db):
        sigma = np.sqrt(0.5 / 10 ** (snr_db / 10))
        s1 = s1 + sigma * rng.standard_normal(n_samples)
        s2 = s2 + sigma * rng.standard_normal(n_samples)
    return t, s1, s2
