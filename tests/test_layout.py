import numpy as np

from radiotel import geometry, layout, sky


def test_points_in_polygon():
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1.0]])
    inside = layout.points_in_polygon(np.array([[0.5, 0.5], [1.5, 0.5], [-0.1, 0.2]]), sq)
    assert inside.tolist() == [True, False, False]


def test_random_layout_respects_constraints(rng):
    pos = layout.random_layout(8, geometry.CAMPUS_POLYGON_ENU, 20.0, rng)
    assert layout.points_in_polygon(pos[:, :2], geometry.CAMPUS_POLYGON_ENU).all()
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1) + np.eye(8) * 1e9
    assert d.min() >= 20.0


def test_fill_fraction_monotone_in_antennas(rng):
    obs = sky.Observation()
    uv_max = layout.max_extent(geometry.CAMPUS_POLYGON_ENU) / obs.wavelength
    pos = layout.random_layout(8, geometry.CAMPUS_POLYGON_ENU, 15.0, rng)
    assert layout.uv_fill_fraction(pos[:4], obs, uv_max) <= layout.uv_fill_fraction(pos, obs, uv_max)


def test_annealing_beats_its_start_and_stays_valid(rng):
    obs = sky.Observation()
    res = layout.anneal_layout(5, obs, n_iter=400, rng=rng)
    uv_max = layout.max_extent(geometry.CAMPUS_POLYGON_ENU) / obs.wavelength
    ha = geometry.hour_angles(obs.duration_s, 24)
    assert res.score >= layout.uv_fill_fraction(res.initial, obs, uv_max, hour_angles=ha)
    assert layout.points_in_polygon(res.positions[:, :2], geometry.CAMPUS_POLYGON_ENU).all()
    assert np.all(np.diff(res.history) >= 0)
