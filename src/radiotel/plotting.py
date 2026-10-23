"""Shared figure style: one palette, fixed colour per entity, recessive chrome.

Colours are the validated reference categorical palette (light mode).  Each
backend / estimator owns a slot permanently, so a chart that shows a subset
never repaints the survivors; every series also has its own marker shape so
identity never depends on colour alone.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e1"
REFERENCE = "#8a8984"  # neutral for reference curves (CRLB, N^2 guide)

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

BACKEND_SLOT = {
    "c-avx2-omp": 0,
    "c-avx2": 1,
    "c-scalar": 2,
    "numpy-gemm": 3,
    "numpy-loop": 4,
    "torch-cpu": 5,
    "torch-cuda": 6,
}
ESTIMATOR_SLOT = {"fft_peak": 0, "hilbert_unwrap": 1, "hilbert_cross": 2, "xcorr_lag": 3}

# Sequential single-hue ramp (surface -> blue 700) for images.
SKY_CMAP = LinearSegmentedColormap.from_list(
    "sky", [SURFACE, "#b7d3f6", "#6da7ec", "#2a78d6", "#1c5cab", "#0d366b"]
)


def style(slot: int, **overrides) -> dict:
    """Line/marker kwargs for a palette slot; keyword overrides win."""
    return {**dict(color=PALETTE[slot], marker=MARKERS[slot], markersize=5, linewidth=1.6), **overrides}


def backend_style(name: str, **overrides) -> dict:
    return style(BACKEND_SLOT[name], **overrides)


def estimator_style(name: str, **overrides) -> dict:
    return style(ESTIMATOR_SLOT[name], **overrides)


def apply_theme() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.labelsize": 9.5,
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "legend.labelcolor": INK,
            "font.size": 9.5,
            "text.color": INK,
            "figure.dpi": 110,
        }
    )


def save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path
