"""Where should the campus receivers go?

Compares, for N = 3..8 receivers, the u-v fill fraction of
* the hand-made 4-receiver layouts from the original notebook (line,
  clustered line, square - scaled to the campus),
* random placements inside the campus footprint (median and best of 200),
* simulated annealing (best of 3 restarts),
and shows the PSF of a random vs an optimised 6-receiver array.
Writes benchmarks/results/layout.csv and figs/layout.png (with --quick, both go to --out).

    python examples/03_optimize_layout.py [--quick]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radiotel import geometry, imaging, layout, plotting, sky  # noqa: E402
from radiotel.plotting import INK_2, PALETTE, SKY_CMAP, plt  # noqa: E402

POLY = geometry.CAMPUS_POLYGON_ENU


def psf_of(pos, obs, npix=128):
    uv = layout.uv_points(pos, obs)
    uvw = np.column_stack([uv, np.zeros(len(uv))])
    cell = 1.0 / (4 * layout.max_extent(POLY) / obs.wavelength)
    return imaging.dirty_beam(uvw, npix, cell, pad=1), cell


def main(quick: bool = False, out: Path = ROOT / "figs") -> None:
    rng = np.random.default_rng(5)
    obs = sky.Observation()
    uv_max = layout.max_extent(POLY) / obs.wavelength
    ha = geometry.hour_angles(obs.duration_s, 24)
    n_random, n_iter, restarts = (40, 400, 1) if quick else (200, 3000, 3)

    def fill(p):
        return layout.uv_fill_fraction(p, obs, uv_max, hour_angles=ha)

    rows, best_layouts = [], {}
    for n in range(3, 9):
        rand = [fill(layout.random_layout(n, POLY, 15.0, rng)) for _ in range(n_random)]
        runs = [layout.anneal_layout(n, obs, n_iter=n_iter, rng=rng) for _ in range(restarts)]
        best = max(runs, key=lambda r: r.score)
        best_layouts[n] = best
        rows.append({"n_ant": n, "n_baselines": n * (n - 1) // 2, "random_median": np.median(rand),
                     "random_best": np.max(rand), "annealed": best.score})
        print(f"N={n}: random median {np.median(rand):.3f}, best {np.max(rand):.3f}, annealed {best.score:.3f}")

    legacy = {name: fill(p * 2.0 - [200, 200, 0]) for name, p in geometry.standard_layouts().items()}
    for name, f in legacy.items():
        print(f"legacy 4-receiver '{name}': {f:.3f}")

    # quick (CI) runs must not overwrite the committed full results
    results = out if quick else ROOT / "benchmarks" / "results"
    results.mkdir(parents=True, exist_ok=True)
    with open(results / "layout.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0]) + ["legacy_line", "legacy_clustered_line", "legacy_square"])
        w.writeheader()
        for r in rows:
            if r["n_ant"] == 4:
                r = {**r, "legacy_line": legacy["line"], "legacy_clustered_line": legacy["clustered line"],
                     "legacy_square": legacy["square"]}
            w.writerow(r)

    plotting.apply_theme()
    fig = plt.figure(figsize=(13, 7.2))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0:2])
    ns = [r["n_ant"] for r in rows]
    series = [("annealed", "annealed", 0), ("random_best", "best of random", 1), ("random_median", "median random", 2)]
    for key, label, slot in series:
        ys = [r[key] for r in rows]
        ax.plot(ns, ys, **plotting.style(slot), label=label)
        ax.annotate(label, (ns[-1], ys[-1]), xytext=(6, 0), textcoords="offset points", va="center",
                    color=INK_2, fontsize=8.5)
    for k, (name, f) in enumerate(legacy.items()):
        ax.plot([4], [f], **plotting.style(3 + k, linestyle="none", markersize=7), label=f"legacy {name}")
    ax.set(title="u-v fill fraction vs number of receivers", xlabel="receivers", ylabel="fraction of u-v cells sampled",
           xlim=(2.7, 9.0))
    ax.legend(loc="upper left")

    ax = fig.add_subplot(gs[0, 2])
    hist = best_layouts[6].history
    ax.plot(np.arange(len(hist)), hist, color=PALETTE[0], linewidth=1.6)
    ax.set(title="Annealing progress (6 receivers)", xlabel="iteration", ylabel="best fill fraction")

    rand6 = layout.random_layout(6, POLY, 15.0, rng)
    opt6 = best_layouts[6].positions
    # colour follows the entity: optimised = annealed (slot 0), random = slot 2
    for k, (pos, title, slot) in enumerate([(rand6, "random", 2), (opt6, "optimised", 0)]):
        ax = fig.add_subplot(gs[1, k])
        poly = np.vstack([POLY, POLY[:1]])
        ax.fill(poly[:, 0], poly[:, 1], color="#eef3fb")
        ax.plot(poly[:, 0], poly[:, 1], color=plotting.REFERENCE, linewidth=1)
        ax.plot(pos[:, 0], pos[:, 1], **plotting.style(slot, linestyle="none", markersize=8))
        ax.set(title=f"{title} layout: fill {fill(pos):.3f}", xlabel="east (m)", ylabel="north (m)", aspect="equal")

    ax = fig.add_subplot(gs[1, 2])
    for pos, label, slot in [(rand6, "random", 2), (opt6, "optimised", 0)]:
        psf, cell = psf_of(pos, obs)
        c = psf.shape[0] // 2
        r_arcmin = np.arange(c) * np.degrees(cell) * 60
        y, x = np.mgrid[: psf.shape[0], : psf.shape[1]] - c
        rad = np.hypot(x, y).astype(int)
        prof = np.array([np.max(np.abs(psf[rad == i])) for i in range(c)])
        ax.plot(r_arcmin, prof, color=PALETTE[slot], linewidth=1.6,
                label=f"{label} (peak sidelobe {imaging.peak_sidelobe(psf):.2f})")
    ax.set(title="PSF: max |sidelobe| vs radius", xlabel="radius (arcmin)", ylabel="max |PSF|", ylim=(0, 1.02))
    ax.legend(loc="upper right")

    path = plotting.save(fig, out / "layout.png")
    print(f"-> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "figs")
    a = ap.parse_args()
    main(a.quick, a.out)
