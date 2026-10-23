import numpy as np

from radiotel import signals


def test_complex_noise_power(rng):
    x = signals.complex_noise(rng, 200_000, power=3.0)
    assert abs(np.mean(np.abs(x) ** 2) - 3.0) < 0.05


def test_quantize_roundtrip(rng):
    x = signals.complex_noise(rng, (2, 5000))
    iq = signals.quantize_cint8(x)
    assert iq.dtype == np.int8 and iq.shape == (2, 10000)
    y = signals.dequantize_cint8(iq)
    corr = np.abs(np.vdot(x.ravel(), y.ravel())) / np.linalg.norm(x) / np.linalg.norm(y)
    assert corr > 0.999
