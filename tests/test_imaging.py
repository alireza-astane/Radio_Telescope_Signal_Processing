import numpy as np

from radiotel import imaging, layout, pipeline, sky
from radiotel.geometry import CAMPUS_POLYGON_ENU


def _setup(sky_model, rng, observe=False):
    obs = sky.Observation(n_snapshots=16)
    pos = layout.random_layout(6, CAMPUS_POLYGON_ENU, 15.0, rng)
    v = pipeline.observe(pos, sky_model, obs, rng=rng) if observe else pipeline.predict(pos, sky_model, obs)
    uvw, vis = v.flat()
    cell = 1.0 / (4 * np.max(np.abs(uvw[:, :2])))
    return uvw, vis, cell


def test_point_source_peaks_at_right_pixel():
    uvw, _, cell = _setup(sky.Sky([sky.PointSource(0, 0, 1.0)]), np.random.default_rng(7))
    src = sky.PointSource(30 * cell, -20 * cell, 1.0)  # exactly on a pixel centre
    uvw, vis, cell = _setup(sky.Sky([src]), np.random.default_rng(7))
    npix = 128
    img = imaging.dirty_image(uvw, vis, npix, cell)
    peak = np.unravel_index(np.argmax(img), img.shape)
    assert peak == imaging.lm_to_pixel(src.l, src.m, npix, cell)
    assert abs(img.max() - 1.0) < 0.05


def test_psf_peak_is_one_and_centered(rng):
    uvw, _, cell = _setup(sky.Sky([sky.PointSource(0, 0, 1)]), rng)
    psf = imaging.dirty_beam(uvw, 64, cell)
    assert psf.shape == (128, 128)
    assert np.unravel_index(np.argmax(psf), psf.shape) == (64, 64)
    assert abs(psf.max() - 1.0) < 1e-9


def test_clean_recovers_fluxes_from_correlated_voltages(rng):
    model = sky.demo_sky()
    uvw, vis, cell = _setup(model, rng, observe=True)
    npix = 256
    dirty = imaging.dirty_image(uvw, vis, npix, cell)
    psf = imaging.dirty_beam(uvw, npix, cell)
    res = imaging.hogbom_clean(dirty, psf, n_iter=3000, threshold=0.01)
    for s in model:
        r, c = imaging.lm_to_pixel(s.l, s.m, npix, cell)
        assert abs(res.model[r - 2 : r + 3, c - 2 : c + 3].sum() - s.flux) < 0.12 * s.flux + 0.03
    assert np.abs(res.residual).max() < np.abs(dirty).max() * 0.1
