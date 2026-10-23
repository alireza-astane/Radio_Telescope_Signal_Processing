"""Two receivers, one source: where the original project started.

Shows (a) the two noisy voltage streams, which look unrelated sample by
sample; (b) their cross-power spectrum, whose phase is the pairwise phase
difference in every channel; and (c) how that phase (the "fringe") rotates
as the Earth turns - the information aperture synthesis turns into an image.

    python examples/01_two_element_fringe.py [--quick]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radiotel import geometry, plotting, signals, sky  # noqa: E402
from radiotel.correlator import correlate  # noqa: E402
from radiotel.plotting import INK_2, PALETTE, plt  # noqa: E402


def main(quick: bool = False, out: Path = ROOT / "figs") -> None:
    rng = np.random.default_rng(3)
    obs = sky.Observation(nchan=64)
    pos = np.array([[0.0, 0.0, 0.0], [120.0, 45.0, 0.0]])  # a 128 m baseline
    src = sky.Sky([sky.PointSource(0.004, 0.002, 1.0)])

    volts = signals.simulate_voltages(pos, src, obs, hour_angle=0.2, n_samples=1 << 15,
                                      noise_power=1.0, rng=rng)
    vis = correlate(volts, obs.nchan)[0]
    freqs_mhz = (obs.channel_freqs() - obs.f0) / 1e6
    order = np.argsort(freqs_mhz)

    has = geometry.hour_angles(obs.duration_s, 24 if quick else 96)
    fringe = np.array([
        correlate(signals.simulate_voltages(pos, src, obs, h, n_samples=1 << 12, rng=rng), obs.nchan)[0].mean()
        for h in has
    ])
    uvw = geometry.uvw(geometry.baselines(pos), has, obs.dec, obs.lat)[:, 0] / obs.wavelength
    model = src.model_visibilities(uvw)

    plotting.apply_theme()
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6), gridspec_kw={"width_ratios": [1.2, 1, 1.2]})

    t_us = np.arange(80) / obs.sample_rate * 1e6
    for k, name in enumerate(["antenna 1", "antenna 2"]):
        ax[0].plot(t_us, volts[k, :80].real + 4 * (1 - k), color=PALETTE[k], linewidth=1.2, label=name)
        ax[0].text(t_us[-1] + 0.4, 4 * (1 - k), name, color=INK_2, va="center", fontsize=8.5)
    ax[0].set(title="Raw voltages (real part)", xlabel="time (µs)", yticks=[])
    ax[0].set_xlim(t_us[0], t_us[-1] + 8)

    ax[1].plot(freqs_mhz[order], np.angle(vis[order]), **plotting.style(0, linestyle="none"))
    ax[1].axhline(np.angle(vis.mean()), color=plotting.REFERENCE, linewidth=1, linestyle="--")
    ax[1].set(title="Cross-spectrum phase", xlabel="baseband frequency (MHz)",
              ylabel="arg V₁₂ (rad)", ylim=(-np.pi, np.pi))

    hours = has * 12 / np.pi
    ax[2].plot(hours, np.angle(model), color=plotting.REFERENCE, linewidth=1, label="model")
    ax[2].plot(hours, np.angle(fringe), **plotting.style(1, linestyle="none"), label="correlated")
    ax[2].set(title="Fringe phase as the Earth rotates", xlabel="hour angle (h)",
              ylabel="arg V₁₂ (rad)", ylim=(-np.pi, np.pi))
    ax[2].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    path = plotting.save(fig, out / "fringe.png")

    err = np.angle(fringe / model)
    print(f"cross-spectrum phase {np.angle(vis.mean()):+.3f} rad; "
          f"fringe-phase rms error vs model {np.sqrt(np.mean(err ** 2)):.3f} rad -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "figs")
    a = ap.parse_args()
    main(a.quick, a.out)
