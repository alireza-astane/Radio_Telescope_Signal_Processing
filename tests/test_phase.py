import numpy as np
import pytest

from radiotel import phase, signals


@pytest.mark.parametrize("name", list(phase.ESTIMATORS))
@pytest.mark.parametrize("true_phi", [-2.5, -0.3, 0.7, 2.9])
def test_noise_free_recovery(name, true_phi):
    _, s1, s2 = signals.two_tone_signals(97.3, 1024.0, 4096, true_phi)
    est = phase.ESTIMATORS[name](s1, s2, 1024.0, 97.3)
    assert abs(phase.wrap(est - true_phi)) < 0.02


def test_xcorr_needs_no_frequency_hint():
    _, s1, s2 = signals.two_tone_signals(50.0, 1000.0, 2000, 1.0)
    assert abs(phase.xcorr_lag(s1, s2, 1000.0) - 1.0) < 0.02


def test_cross_spectrum_estimator_approaches_crlb(rng):
    n, snr_db = 1024, 10.0
    errs = []
    for _ in range(400):
        _, s1, s2 = signals.two_tone_signals(97.3, 1024.0, n, 1.0, snr_db, rng)
        errs.append(phase.wrap(phase.hilbert_cross(s1, s2) - 1.0))
    bound = phase.crlb_phase_difference(n, 10 ** (snr_db / 10))
    assert np.var(errs) < 1.5 * bound
    assert abs(np.mean(errs)) < 4 * np.sqrt(bound / len(errs)) + 1e-3


def test_wrap():
    np.testing.assert_allclose(phase.wrap([0, np.pi, -np.pi, 3 * np.pi, 7.0]), [0, np.pi, np.pi, np.pi, 7.0 - 2 * np.pi])
