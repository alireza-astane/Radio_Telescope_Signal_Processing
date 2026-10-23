"""Array geometry: antenna positions, baselines and (u, v, w) coordinates.

Conventions
-----------
* Antenna positions are local East-North-Up (ENU) offsets in metres, shape
  ``(n_ant, 3)``.
* A baseline joins antennas ``i < j`` and points from ``i`` to ``j``:
  ``b_ij = r_j - r_i``.  Pairs are ordered as ``np.triu_indices(n_ant, 1)``,
  so there are ``n_ant * (n_ant - 1) / 2`` of them.
* The visibility of pair (i, j) is ``V_ij = <x_i x_j^*>`` which, for a sky
  brightness ``I(l, m)``, equals ``sum I exp(-2 pi i (u l + v m + w (n - 1)))``
  with ``(u, v, w) = b_ij / lambda`` expressed in the frame tracking the
  phase centre.
"""

from __future__ import annotations

import numpy as np

C_LIGHT = 299_792_458.0  # m/s
EARTH_RADIUS = 6_371_000.0  # m
SIDEREAL_DAY = 86_164.0905  # s

# Sharif University of Technology, Tehran (approximate campus centre).
SHARIF_LAT_DEG = 35.7036
SHARIF_LON_DEG = 51.3517

# An *illustrative* campus footprint in local ENU metres around the centre
# above.  It is a rough outline used to constrain the layout optimiser, not a
# survey of real receiver sites.
CAMPUS_POLYGON_ENU = np.array(
    [
        [-300.0, -220.0],
        [280.0, -220.0],
        [300.0, 60.0],
        [180.0, 230.0],
        [-260.0, 230.0],
        [-310.0, 20.0],
    ]
)


def latlon_to_enu(lat_deg, lon_deg, lat0_deg=SHARIF_LAT_DEG, lon0_deg=SHARIF_LON_DEG):
    """Convert latitude/longitude to local ENU metres (small-area tangent plane)."""
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    lon = np.radians(np.asarray(lon_deg, dtype=float))
    lat0, lon0 = np.radians(lat0_deg), np.radians(lon0_deg)
    east = EARTH_RADIUS * np.cos(lat0) * (lon - lon0)
    north = EARTH_RADIUS * (lat - lat0)
    return np.stack([east, north, np.zeros_like(east)], axis=-1)


def baseline_pairs(n_ant: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices ``(i, j)`` with ``i < j`` for every baseline of an ``n_ant`` array."""
    return np.triu_indices(n_ant, k=1)


def baselines(positions: np.ndarray) -> np.ndarray:
    """All baseline vectors ``r_j - r_i`` (``i < j``), shape ``(n_bl, 3)``."""
    positions = np.asarray(positions, dtype=float)
    i, j = baseline_pairs(len(positions))
    return positions[j] - positions[i]


def enu_to_xyz(enu: np.ndarray, lat_rad: float) -> np.ndarray:
    """Rotate ENU vectors to the local equatorial frame (X to meridian, Z to pole)."""
    e, n, u = np.moveaxis(np.asarray(enu, dtype=float), -1, 0)
    sl, cl = np.sin(lat_rad), np.cos(lat_rad)
    x = -sl * n + cl * u
    y = e
    z = cl * n + sl * u
    return np.stack([x, y, z], axis=-1)


def uvw_matrix(hour_angle, dec_rad):
    """Rotation(s) from equatorial XYZ to (u, v, w) towards ``(H, dec)``.

    ``hour_angle`` may be an array; the result has shape ``(..., 3, 3)``.
    This is the matrix built cell-by-cell in the original ``getV`` notebook
    code (legacy/methods.ipynb).
    """
    h = np.asarray(hour_angle, dtype=float)
    sh, ch = np.sin(h), np.cos(h)
    sd, cd = np.sin(dec_rad), np.cos(dec_rad)
    zero = np.zeros_like(h)
    return np.stack(
        [
            np.stack([sh, ch, zero], axis=-1),
            np.stack([-sd * ch, sd * sh, np.full_like(h, cd)], axis=-1),
            np.stack([cd * ch, -cd * sh, np.full_like(h, sd)], axis=-1),
        ],
        axis=-2,
    )


def uvw(enu_vectors, hour_angle, dec_rad, lat_rad=np.radians(SHARIF_LAT_DEG)):
    """Project ENU vectors (antennas or baselines) to (u, v, w) in metres.

    Returns shape ``(n_ha, n_vec, 3)`` for ``hour_angle`` of shape ``(n_ha,)``
    (or ``(n_vec, 3)`` for a scalar hour angle).
    """
    xyz = enu_to_xyz(enu_vectors, lat_rad)
    m = uvw_matrix(hour_angle, dec_rad)
    return np.einsum("...ab,nb->...na", m, xyz)


def hour_angles(duration_s: float, n_steps: int, ha0_rad: float = 0.0) -> np.ndarray:
    """Hour angles of an observation of ``duration_s`` centred on ``ha0_rad``."""
    span = 2 * np.pi * duration_s / SIDEREAL_DAY
    return ha0_rad + np.linspace(-span / 2, span / 2, n_steps)


def elevation(hour_angle, dec_rad, lat_rad=np.radians(SHARIF_LAT_DEG)):
    """Source elevation (rad); useful to discard below-horizon samples."""
    h = np.asarray(hour_angle, dtype=float)
    s = np.sin(lat_rad) * np.sin(dec_rad) + np.cos(lat_rad) * np.cos(dec_rad) * np.cos(h)
    return np.arcsin(np.clip(s, -1.0, 1.0))


def standard_layouts(scale: float = 200.0) -> dict[str, np.ndarray]:
    """The hand-made 4-antenna layouts from the original experiments, in metres."""
    def enu(xy):
        xy = np.asarray(xy, dtype=float) * scale
        return np.column_stack([xy, np.zeros(len(xy))])

    return {
        "line": enu([[0, 0], [0.333, 0.333], [0.666, 0.666], [1, 1]]),
        "clustered line": enu([[0, 0], [0.49, 0.49], [0.51, 0.51], [1, 1]]),
        "square": enu([[0, 0], [0, 1], [1, 0], [1, 1]]),
    }
