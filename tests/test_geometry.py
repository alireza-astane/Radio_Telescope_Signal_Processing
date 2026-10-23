import numpy as np
import pytest

from radiotel import geometry as g


@pytest.mark.parametrize("n", [2, 3, 4, 6, 10])
def test_baseline_count_is_n_choose_2(n):
    pos = np.random.default_rng(n).normal(size=(n, 3))
    assert len(g.baselines(pos)) == n * (n - 1) // 2


def test_baselines_generalise_legacy_get_Bs():
    pos = g.standard_layouts(scale=1.0)["square"]
    # legacy get_Bs(poses)[:, k] = poses[:, j] - poses[:, i] for pairs (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)
    legacy = np.array([pos[j] - pos[i] for i in range(4) for j in range(i + 1, 4)])
    np.testing.assert_allclose(g.baselines(pos), legacy)


def test_uvw_rotation_preserves_length(rng):
    b = rng.normal(size=(5, 3)) * 100
    out = g.uvw(b, np.linspace(-1, 1, 7), np.radians(40))
    np.testing.assert_allclose(np.linalg.norm(out, axis=-1), np.broadcast_to(np.linalg.norm(b, axis=-1), (7, 5)))


def test_uvw_known_case_source_at_pole():
    # East-west baseline, source at the celestial pole: the baseline rotates in the
    # u-v plane as the Earth turns (u along it at transit, v along it 6 h later), w = 0.
    b = np.array([[100.0, 0.0, 0.0]])
    h = np.array([0.0, np.pi / 2])
    out = g.uvw(b, h, np.pi / 2, lat_rad=np.radians(35))
    np.testing.assert_allclose(out[0, 0], [100.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(out[1, 0], [0.0, 100.0, 0.0], atol=1e-9)


def test_east_west_track_is_ellipse():
    b = np.array([[200.0, 0.0, 0.0]])
    dec = np.radians(30)
    uv = g.uvw(b, np.linspace(0, 2 * np.pi, 100), dec)[:, 0, :2]
    # u^2 + (v / sin dec)^2 = |b|^2 for an E-W baseline
    np.testing.assert_allclose(uv[:, 0] ** 2 + (uv[:, 1] / np.sin(dec)) ** 2, 200.0**2)


def test_latlon_to_enu_roundtrip_scale():
    enu = g.latlon_to_enu(g.SHARIF_LAT_DEG + 0.001, g.SHARIF_LON_DEG)
    assert abs(enu[1] - 111.2) < 0.5 and abs(enu[0]) < 1e-9
