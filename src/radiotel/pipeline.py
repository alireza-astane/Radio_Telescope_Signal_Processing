"""End-to-end observation: voltages -> correlator -> visibilities on u-v tracks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import correlator
from .geometry import C_LIGHT, baselines, hour_angles as make_hour_angles, uvw
from .signals import simulate_voltages
from .sky import Observation, Sky


@dataclass
class Visibilities:
    vis: np.ndarray  # (n_ha, n_bl, nchan) complex
    uvw_m: np.ndarray  # (n_ha, n_bl, 3) baseline coordinates in metres
    freqs: np.ndarray  # (nchan,) absolute frequency of each channel (Hz)
    positions: np.ndarray  # (n_ant, 3) ENU metres
    hour_angles: np.ndarray  # (n_ha,)

    @property
    def n_ant(self) -> int:
        return len(self.positions)

    def uvw_lambda(self) -> np.ndarray:
        """Per-channel (u, v, w) in wavelengths, shape ``(n_ha, n_bl, nchan, 3)``."""
        return self.uvw_m[:, :, None, :] * (self.freqs / C_LIGHT)[None, None, :, None]

    def flat(self) -> tuple[np.ndarray, np.ndarray]:
        """All samples as ``(uvw_lambda (N, 3), vis (N,))`` for imaging."""
        return self.uvw_lambda().reshape(-1, 3), self.vis.reshape(-1)

    def with_vis(self, vis: np.ndarray) -> "Visibilities":
        return Visibilities(vis, self.uvw_m, self.freqs, self.positions, self.hour_angles)


def default_backend() -> str:
    available = correlator.available_backends()
    for b in ("c-avx2-omp", "numpy-gemm"):
        if b in available:
            return b
    return available[0]


def _tracks(positions, obs: Observation, hour_angles):
    if hour_angles is None:
        hour_angles = make_hour_angles(obs.duration_s, obs.n_snapshots)
    return np.asarray(hour_angles), uvw(baselines(positions), hour_angles, obs.dec, obs.lat)


def observe(
    positions,
    sky: Sky,
    obs: Observation,
    hour_angles=None,
    noise_power: float = 1.0,
    gains=None,
    backend: str | None = None,
    rng: np.random.Generator | None = None,
) -> Visibilities:
    """Simulate voltages at every snapshot and correlate them (the full pipeline)."""
    rng = rng or np.random.default_rng()
    backend = backend or default_backend()
    has, uvw_m = _tracks(positions, obs, hour_angles)
    vis = np.empty((len(has), uvw_m.shape[1], obs.nchan), dtype=np.complex64)
    for k, ha in enumerate(has):
        volts = simulate_voltages(positions, sky, obs, ha, noise_power=noise_power, gains=gains, rng=rng)
        vis[k] = correlator.correlate(volts, obs.nchan, backend)
    return Visibilities(vis, uvw_m, obs.channel_freqs(), np.asarray(positions, float), has)


def predict(positions, sky: Sky, obs: Observation, hour_angles=None) -> Visibilities:
    """Noise-free analytic visibilities (fast; used for layout studies and as a model)."""
    has, uvw_m = _tracks(positions, obs, hour_angles)
    out = Visibilities(None, uvw_m, obs.channel_freqs(), np.asarray(positions, float), has)
    out.vis = sky.model_visibilities(out.uvw_lambda())
    return out
