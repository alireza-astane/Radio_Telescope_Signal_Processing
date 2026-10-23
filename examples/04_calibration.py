"""Receiver gain calibration and closure phases.

Real receivers add unknown phases (cable lengths, oscillators) and gains.
Here every receiver gets a random phase error of ~1 rad rms and 10 % gain
error.  The corrupted target image is unusable; a short observation of a
calibrator source at the phase centre lets StEFCal solve the N antenna gains
from the N(N-1)/2 baselines, after which the image is restored.  Closure
phases, sums around baseline triangles, are unaffected by the errors.

    python examples/04_calibration.py [--quick]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from matplotlib.colors import PowerNorm  # noqa: E402

from radiotel import calibration, geometry, imaging, layout, pipeline, plotting, signals, sky  # noqa: E402
from radiotel.plotting import INK_2, PALETTE, SKY_CMAP, plt  # noqa: E402


def main(quick: bool = False, out: Path = ROOT / "figs") -> None:
    rng = np.random.default_rng(21)
    n_ant = 6
    obs = sky.Observation(n_snapshots=16 if quick else 40)
    pos = layout.random_layout(n_ant, geometry.CAMPUS_POLYGON_ENU, 15.0, rng)
    gains = signals.random_gains(n_ant, phase_rms_rad=1.0, amp_rms=0.1, rng=rng)

    # 1. Calibrator scan: a single bright source at the phase centre, short and few snapshots.
    cal_obs = sky.Observation(n_snapshots=4, duration_s=1800.0)
    cal = pipeline.observe(pos, sky.Sky([sky.PointSource(0, 0, 5.0)]), cal_obs, gains=gains, rng=rng)
    sol = calibration.stefcal(np.moveaxis(cal.vis, 1, -1), 5.0, n_ant)
    true = gains * np.exp(-1j * np.angle(gains[0]))
    phase_err = np.angle(sol / true)

    # 2. Target observation with the same receivers, corrupted by the same gains.
    target = pipeline.observe(pos, sky.demo_sky(), obs, gains=gains, rng=rng)
    raw_bl = np.moveaxis(target.vis, 1, -1)  # (n_ha, nchan, n_bl)
    fixed = target.with_vis(np.moveaxis(calibration.apply_gains(raw_bl, sol), -1, 1))

    # Closure phases from channel-averaged visibilities (the phase changes by
    # < 0.1 deg across the 2.4 MHz band), raw corrupted data vs the true sky.
    model = pipeline.predict(pos, sky.demo_sky(), obs)
    raw_avg = raw_bl.mean(axis=1)  # (n_ha, n_bl)
    true_avg = np.moveaxis(model.vis, 1, -1).mean(axis=1)
    cp_raw, tris = calibration.closure_phases(raw_avg, n_ant)
    cp_true, _ = calibration.closure_phases(true_avg, n_ant)
    cp_err = np.degrees(np.abs(np.angle(np.exp(1j * (cp_raw - cp_true))))).mean(axis=0)
    bl_err = np.degrees(np.abs(np.angle(raw_avg * np.conj(true_avg)))).mean(axis=0)

    uvw, _ = target.flat()
    npix, cell = 192, 1.0 / (4 * np.max(np.abs(uvw[:, :2])))
    psf = imaging.dirty_beam(uvw, npix, cell)
    imgs = {}
    for label, v in [("before calibration", target), ("after calibration", fixed)]:
        dirty = imaging.dirty_image(uvw, v.flat()[1], npix, cell)
        imgs[label] = imaging.hogbom_clean(dirty, psf, n_iter=1500, threshold=0.03)

    plotting.apply_theme()
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.8), gridspec_kw={"width_ratios": [1, 1, 1, 1.1]})

    k = np.arange(n_ant)
    ax[0].bar(k - 0.2, np.angle(true), width=0.38, color=PALETTE[1], label="injected")
    ax[0].bar(k + 0.2, np.angle(sol), width=0.38, color=PALETTE[0], label="solved")
    ax[0].set(title="Receiver phase errors", xlabel="receiver", ylabel="phase (rad)",
              xticks=k, xticklabels=[f"R{i + 1}" for i in k])
    ax[0].legend(loc="lower left")
    ax[0].text(0.98, 0.98, f"max error {np.max(np.abs(phase_err)):.3f} rad", transform=ax[0].transAxes,
               ha="right", va="top", color=INK_2, fontsize=8.5)

    ext = np.degrees(imaging.image_extent(npix, cell)) * 60
    norm = PowerNorm(gamma=0.5, vmin=0, vmax=1.0)
    for a, (label, res) in zip(ax[1:3], imgs.items()):
        a.imshow(np.clip(res.restored, 0, None), origin="lower", extent=ext, cmap=SKY_CMAP, norm=norm)
        a.set(title=f"CLEAN image {label}", xlabel="l (arcmin)", ylabel="m (arcmin)", xlim=(-32, 32), ylim=(-32, 32))
        a.grid(False)

    ax[3].plot(np.arange(len(bl_err)), bl_err, **plotting.style(1), label="baseline phase")
    ax[3].plot(np.arange(len(cp_err)), cp_err, **plotting.style(0), label="closure phase")
    ax[3].set(title="Raw data: phase error vs true sky", xlabel="baseline / triangle index",
              ylabel="mean |error| over 8 h (deg)")
    ax[3].legend(loc="upper right")
    fig.tight_layout()
    path = plotting.save(fig, out / "calibration.png")

    print(f"solved gains: max phase error {np.max(np.abs(phase_err)):.4f} rad, "
          f"max amplitude error {np.max(np.abs(np.abs(sol) / np.abs(true) - 1)):.4f}")
    print(f"uncalibrated data vs true sky: baseline phases off by {bl_err.mean():.1f} deg on average, "
          f"closure phases by {cp_err.mean():.2f} deg (noise only)")
    print(f"-> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "figs")
    a = ap.parse_args()
    main(a.quick, a.out)
