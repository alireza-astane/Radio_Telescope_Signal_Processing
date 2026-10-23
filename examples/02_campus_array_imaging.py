"""End-to-end campus interferometer: from raw voltages to a CLEANed image.

1. place 6 receivers on the (illustrative) campus footprint, optimised for
   u-v coverage;
2. for 48 snapshots over 8 hours, simulate every receiver's voltages for a
   four-source sky and correlate them (FX correlator, C/AVX2 backend);
3. grid the visibilities on their Earth-rotation u-v tracks, make the dirty
   image and dirty beam, and deconvolve with Högbom CLEAN.

    python examples/02_campus_array_imaging.py [--quick]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radiotel import geometry, imaging, layout, pipeline, plotting, sky  # noqa: E402
from matplotlib.colors import PowerNorm  # noqa: E402

from radiotel.plotting import INK_2, PALETTE, SKY_CMAP, plt  # noqa: E402


def main(quick: bool = False, out: Path = ROOT / "figs") -> None:
    rng = np.random.default_rng(11)
    obs = sky.Observation(n_snapshots=16 if quick else 48)
    model = sky.demo_sky()

    pos = layout.anneal_layout(6, obs, n_iter=300 if quick else 3000, rng=rng).positions
    t0 = time.perf_counter()
    vis = pipeline.observe(pos, model, obs, noise_power=4.0, rng=rng)
    t_obs = time.perf_counter() - t0

    uvw, v = vis.flat()
    umax = np.max(np.abs(uvw[:, :2]))
    npix = 256
    cell = 1.0 / (4 * umax)
    dirty = imaging.dirty_image(uvw, v, npix, cell)
    psf = imaging.dirty_beam(uvw, npix, cell)
    res = imaging.hogbom_clean(dirty, psf, n_iter=3000, threshold=0.02)

    # True sky convolved with the same restoring beam, for a like-for-like panel.
    truth = np.zeros_like(dirty)
    for s in model:
        truth[imaging.lm_to_pixel(s.l, s.m, npix, cell)] += s.flux
    truth_img = imaging.restore(truth, imaging.beam_fwhm_pixels(psf))

    plotting.apply_theme()
    fig = plt.figure(figsize=(13, 7.4))
    gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.28)

    ax = fig.add_subplot(gs[0, 0])
    poly = np.vstack([geometry.CAMPUS_POLYGON_ENU, geometry.CAMPUS_POLYGON_ENU[:1]])
    ax.fill(poly[:, 0], poly[:, 1], color="#eef3fb", zorder=0)
    ax.plot(poly[:, 0], poly[:, 1], color=plotting.REFERENCE, linewidth=1)
    ax.plot(pos[:, 0], pos[:, 1], **plotting.style(0, linestyle="none", markersize=8))
    for k, p in enumerate(pos):
        ax.annotate(f"R{k + 1}", p[:2], xytext=(5, 5), textcoords="offset points", color=INK_2, fontsize=8.5)
    ax.set(title="Receiver layout (campus footprint)", xlabel="east (m)", ylabel="north (m)", aspect="equal")

    ax = fig.add_subplot(gs[0, 1])
    uvk = uvw[:, :2] / 1e3
    ax.scatter(uvk[:, 0], uvk[:, 1], s=0.4, color=PALETTE[0], linewidths=0)
    ax.scatter(-uvk[:, 0], -uvk[:, 1], s=0.4, color=PALETTE[1], linewidths=0)
    ax.set(title=f"u-v coverage ({vis.vis.shape[1]} baselines × 8 h)", xlabel="u (kλ)", ylabel="v (kλ)",
           aspect="equal")
    ax.text(0.02, 0.02, "blue: measured  ·  orange: conjugate", transform=ax.transAxes, color=INK_2, fontsize=8)

    ext = np.degrees(imaging.image_extent(npix, cell)) * 60  # arcmin
    c = psf.shape[0] // 2
    panels = [
        (gs[0, 2], psf[c - npix // 2 : c + npix // 2, c - npix // 2 : c + npix // 2], "Dirty beam (PSF)"),
        (gs[1, 0], dirty, "Dirty image"),
        (gs[1, 1], res.restored, f"CLEAN restored ({res.n_iter} iterations)"),
        (gs[1, 2], truth_img, "True sky (same restoring beam)"),
    ]
    # Square-root stretch so the 0.3-flux source is visible next to the 1.0 one;
    # negative sidelobes are clipped to the background colour.
    norm = PowerNorm(gamma=0.5, vmin=0.0, vmax=truth_img.max())
    for spec, img, title in panels:
        ax = fig.add_subplot(spec)
        im = ax.imshow(np.clip(img, 0, None), origin="lower", extent=ext, cmap=SKY_CMAP, norm=norm)
        half = 12 if "PSF" in title else 32  # arcmin
        ax.set(title=title, xlabel="l (arcmin)", ylabel="m (arcmin)", xlim=(-half, half), ylim=(-half, half))
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)

    path = plotting.save(fig, out / "imaging.png")

    print(f"observed {vis.vis.shape[0]} snapshots x {vis.vis.shape[1]} baselines x {obs.nchan} channels "
          f"in {t_obs:.1f} s (voltage simulation + correlation)")
    print(f"beam FWHM {imaging.beam_fwhm_pixels(psf) * np.degrees(cell) * 60:.2f} arcmin, "
          f"peak sidelobe {imaging.peak_sidelobe(psf):.2f}")
    print("source   true   CLEAN")
    for s in model:
        r, cc = imaging.lm_to_pixel(s.l, s.m, npix, cell)
        print(f"({s.l:+.3f},{s.m:+.3f})  {s.flux:.2f}   {res.model[r - 2:r + 3, cc - 2:cc + 3].sum():.2f}")
    print(f"-> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "figs")
    a = ap.parse_args()
    main(a.quick, a.out)
