"""Array-layout optimisation: where on campus should the receivers go?

The original notebook scored a layout by plotting its u-v tracks with
matplotlib and counting dark pixels.  Here the same idea is computed
directly: the *u-v fill fraction* is the share of cells of a square u-v
grid (inside the disk the site can reach) that are visited by at
least one baseline during the observation.  More filled cells means a
cleaner point-spread function.  Simulated annealing moves one antenna at a
time inside the site polygon, with a minimum spacing between antennas.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import CAMPUS_POLYGON_ENU, baselines, hour_angles as make_hour_angles, uvw
from .sky import Observation


def points_in_polygon(xy: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Vectorised ray-casting test; ``xy`` is ``(n, 2)``, ``poly`` is ``(k, 2)``."""
    x, y = xy[:, 0:1], xy[:, 1:2]
    x1, y1 = poly[:, 0], poly[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
    crosses = ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-300) + x1)
    return np.count_nonzero(crosses, axis=1) % 2 == 1


def max_extent(poly: np.ndarray) -> float:
    """Longest distance between two polygon vertices (longest possible baseline)."""
    d = poly[:, None, :] - poly[None, :, :]
    return float(np.max(np.hypot(d[..., 0], d[..., 1])))


def uv_points(positions, obs: Observation, hour_angles=None) -> np.ndarray:
    """u-v samples in wavelengths at the band centre, both signs, shape ``(N, 2)``."""
    if hour_angles is None:
        hour_angles = make_hour_angles(obs.duration_s, obs.n_snapshots)
    uv = uvw(baselines(positions), hour_angles, obs.dec, obs.lat)[..., :2].reshape(-1, 2)
    uv = uv / obs.wavelength
    return np.concatenate([uv, -uv])


def uv_fill_fraction(positions, obs: Observation, uv_max: float, n_cells: int = 64,
                     hour_angles=None) -> float:
    """Fraction of u-v cells within radius ``uv_max`` (wavelengths) that are sampled."""
    uv = uv_points(positions, obs, hour_angles)
    idx = np.floor((uv / uv_max + 1.0) * 0.5 * n_cells).astype(int)
    ok = np.all((idx >= 0) & (idx < n_cells), axis=1)
    hit = np.zeros((n_cells, n_cells), dtype=bool)
    hit[idx[ok, 1], idx[ok, 0]] = True
    c = (np.arange(n_cells) + 0.5) / n_cells * 2 - 1
    disk = np.hypot(*np.meshgrid(c, c)) <= 1.0
    return float(np.count_nonzero(hit & disk) / np.count_nonzero(disk))


def random_layout(n_ant: int, poly: np.ndarray, min_sep: float, rng: np.random.Generator,
                  max_tries: int = 100_000) -> np.ndarray:
    """Uniformly random positions inside ``poly`` respecting ``min_sep``."""
    lo, hi = poly.min(axis=0), poly.max(axis=0)
    pts: list[np.ndarray] = []
    for _ in range(max_tries):
        p = rng.uniform(lo, hi)
        if not points_in_polygon(p[None], poly)[0]:
            continue
        if all(np.hypot(*(p - q)) >= min_sep for q in pts):
            pts.append(p)
            if len(pts) == n_ant:
                return np.column_stack([np.array(pts), np.zeros(n_ant)])
    raise RuntimeError("could not place antennas; reduce min_sep or n_ant")


def _valid(xy: np.ndarray, k: int, poly: np.ndarray, min_sep: float) -> bool:
    if not points_in_polygon(xy[k : k + 1], poly)[0]:
        return False
    d = np.hypot(*(xy - xy[k]).T)
    d[k] = np.inf
    return bool(np.all(d >= min_sep))


@dataclass
class AnnealResult:
    positions: np.ndarray
    score: float
    history: list[float] = field(default_factory=list)  # best score per iteration
    initial: np.ndarray | None = None


def anneal_layout(n_ant: int, obs: Observation, poly: np.ndarray = CAMPUS_POLYGON_ENU,
                  n_iter: int = 3000, min_sep: float = 15.0, step0: float | None = None,
                  t0: float = 0.02, n_cells: int = 64, hour_angles=None,
                  rng: np.random.Generator | None = None) -> AnnealResult:
    """Maximise the u-v fill fraction by simulated annealing."""
    rng = rng or np.random.default_rng()
    if hour_angles is None:
        hour_angles = make_hour_angles(obs.duration_s, 24)
    uv_max = max_extent(poly) / obs.wavelength
    step0 = step0 or 0.25 * max_extent(poly)

    def score(p):
        return uv_fill_fraction(p, obs, uv_max, n_cells, hour_angles)

    pos = random_layout(n_ant, poly, min_sep, rng)
    initial = pos.copy()
    cur = best = score(pos)
    best_pos = pos.copy()
    history = []
    for it in range(n_iter):
        frac = 1.0 - it / n_iter
        temp = t0 * frac + 1e-6
        k = rng.integers(n_ant)
        trial = pos.copy()
        trial[k, :2] += rng.normal(scale=step0 * frac + 2.0, size=2)
        if _valid(trial[:, :2], k, poly, min_sep):
            s = score(trial)
            if s >= cur or rng.random() < np.exp((s - cur) / temp):
                pos, cur = trial, s
                if s > best:
                    best, best_pos = s, trial.copy()
        history.append(best)
    return AnnealResult(best_pos, best, history, initial)
