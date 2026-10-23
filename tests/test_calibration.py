import numpy as np

from radiotel import calibration, pipeline, signals, sky
from radiotel.geometry import standard_layouts


def test_stefcal_recovers_injected_gains(rng):
    obs = sky.Observation(n_snapshots=8)
    pos = np.vstack([standard_layouts()["square"], [[90.0, 30.0, 0.0]]])
    gains = signals.random_gains(len(pos), phase_rms_rad=1.0, amp_rms=0.1, rng=rng)
    cal_src = sky.Sky([sky.PointSource(0, 0, 1.0)])
    v = pipeline.observe(pos, cal_src, obs, gains=gains, rng=rng)
    sol = calibration.stefcal(np.moveaxis(v.vis, 1, -1), 1.0, len(pos))
    true = gains * np.exp(-1j * np.angle(gains[0]))
    assert np.max(np.abs(np.angle(sol / true))) < 0.01
    assert np.max(np.abs(np.abs(sol) / np.abs(true) - 1)) < 0.01


def test_closure_phase_is_gain_invariant(rng):
    obs = sky.Observation(n_snapshots=4)
    pos = np.vstack([standard_layouts()["square"]])
    model = sky.demo_sky()
    clean = pipeline.predict(pos, model, obs)
    gains = signals.random_gains(4, 2.0, rng=rng)
    corrupted = calibration.apply_gains(np.moveaxis(clean.vis, 1, -1), gains, invert=False)
    cp0, tris = calibration.closure_phases(np.moveaxis(clean.vis, 1, -1), 4)
    cp1, _ = calibration.closure_phases(corrupted, 4)
    assert len(tris) == 4
    np.testing.assert_allclose(np.angle(np.exp(1j * (cp1 - cp0))), 0, atol=1e-9)


def test_apply_gains_roundtrip(rng):
    vis = rng.normal(size=(3, 6)) + 1j * rng.normal(size=(3, 6))
    g = signals.random_gains(4, 1.0, 0.2, rng)
    back = calibration.apply_gains(calibration.apply_gains(vis, g, invert=False), g)
    np.testing.assert_allclose(back, vis)
