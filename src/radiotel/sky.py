"""Observation parameters and a point-source sky model."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import C_LIGHT, SHARIF_LAT_DEG


@dataclass
class Observation:
    """Parameters of a simulated observation.

    The defaults describe a plausible small campus array: the 21 cm neutral
    hydrogen line observed with a complex-baseband receiver of SDR-class
    bandwidth.  They are assumptions, not specifications of the real Sharif
    receivers.
    """

    f0: float = 1420.40575e6  # RF centre frequency (Hz)
    sample_rate: float = 2.4e6  # complex samples/s = processed bandwidth (Hz)
    dec_deg: float = 40.0  # phase-centre declination
    lat_deg: float = SHARIF_LAT_DEG
    duration_s: float = 8 * 3600.0  # Earth-rotation synthesis span
    n_snapshots: int = 48  # correlator dumps across the span
    samples_per_snapshot: int = 16_384
    nchan: int = 64

    @property
    def dec(self) -> float:
        return np.radians(self.dec_deg)

    @property
    def lat(self) -> float:
        return np.radians(self.lat_deg)

    @property
    def wavelength(self) -> float:
        return C_LIGHT / self.f0

    def channel_freqs(self, nchan: int | None = None) -> np.ndarray:
        """Absolute sky frequency of each correlator channel (FFT order)."""
        nchan = nchan or self.nchan
        return self.f0 + np.fft.fftfreq(nchan, 1.0 / self.sample_rate)


@dataclass
class PointSource:
    l: float  # direction cosine offset from the phase centre (east)
    m: float  # direction cosine offset (north)
    flux: float = 1.0  # power per complex sample, arbitrary units

    @property
    def n(self) -> float:
        return float(np.sqrt(1.0 - self.l**2 - self.m**2))


@dataclass
class Sky:
    sources: list[PointSource] = field(default_factory=list)

    def __iter__(self):
        return iter(self.sources)

    def __len__(self):
        return len(self.sources)

    @property
    def total_flux(self) -> float:
        return float(sum(s.flux for s in self.sources))

    def model_visibilities(self, uvw_lambda: np.ndarray) -> np.ndarray:
        """Analytic visibilities for ``uvw_lambda`` (``(..., 3)``, in wavelengths)."""
        u, v, w = np.moveaxis(uvw_lambda, -1, 0)
        vis = np.zeros(u.shape, dtype=complex)
        for s in self.sources:
            vis += s.flux * np.exp(-2j * np.pi * (u * s.l + v * s.m + w * (s.n - 1.0)))
        return vis


def demo_sky() -> Sky:
    """Four point sources within 0.5 deg of the phase centre.

    That is well inside the ~7 deg primary beam of a 2 m dish at 21 cm and
    small enough that the (ignored) w-term stays negligible.
    """
    return Sky(
        [
            PointSource(0.0, 0.0, 1.0),
            PointSource(0.006, 0.003, 0.6),
            PointSource(-0.004, 0.007, 0.4),
            PointSource(-0.007, -0.005, 0.3),
        ]
    )
