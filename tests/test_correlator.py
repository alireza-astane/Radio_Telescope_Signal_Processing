import numpy as np
import pytest

from radiotel import correlator, geometry, signals, sky
from radiotel.correlator import c_fx


@pytest.fixture
def volts(rng):
    obs = sky.Observation()
    pos = np.array([[0, 0, 0], [35, 5, 0], [-20, 40, 0], [60, -30, 0], [10, 90, 0.0]])
    return signals.simulate_voltages(pos, sky.demo_sky(), obs, 0.4, n_samples=1 << 13, rng=rng)


@pytest.mark.parametrize("backend", correlator.available_backends())
@pytest.mark.parametrize("nchan", [64, 100])  # 100: exercises the AVX tail loop
def test_backends_match_reference(volts, backend, nchan):
    ref = correlator.correlate(volts.astype(np.complex128), nchan, "numpy-loop")
    out = correlator.correlate(volts, nchan, backend)
    assert out.shape == (10, nchan)
    assert np.max(np.abs(out - ref)) / np.max(np.abs(ref)) < 1e-4


def test_visibility_matches_analytic_model(rng):
    obs = sky.Observation()
    pos = np.array([[0, 0, 0], [30, 0, 0], [0, 40, 0], [25, 35, 0.0]])
    src = sky.Sky([sky.PointSource(0.01, 0.005, 1.0)])
    v = signals.simulate_voltages(pos, src, obs, 0.3, n_samples=1 << 16, noise_power=0.1, rng=rng)
    vis = correlator.correlate(v, 64)
    uvw = geometry.uvw(geometry.baselines(pos), 0.3, obs.dec, obs.lat)
    model = src.model_visibilities(uvw[:, None, :] * obs.channel_freqs(64)[None, :, None] / geometry.C_LIGHT)
    assert np.sqrt(np.mean(np.angle(vis / model) ** 2)) < 0.05
    assert abs(np.mean(np.abs(vis)) - 1.0) < 0.05  # unitary F-step: |V| = flux


@pytest.mark.skipif(not c_fx.available(), reason="libfxcorr.so not built")
def test_c_fft_matches_numpy(rng):
    x = signals.complex_noise(rng, (3, 256))
    np.testing.assert_allclose(c_fx.fft(x), np.fft.fft(x), rtol=0, atol=1e-4)
