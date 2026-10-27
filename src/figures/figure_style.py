#!/usr/bin/env python3
"""Shared publication style for OdorBlock figures."""

from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

# Final panel size for a 2 x 2 layout in a two-column figure (~175 mm wide).
PANEL_SIZE = (3.35, 2.65)  # inches ~= 85 x 67 mm

# Font sizes at final publication size.
BASE_FONT_SIZE = 8.0
AXIS_LABEL_SIZE = 8.0
TICK_LABEL_SIZE = 7.0
LEGEND_SIZE = 7.0
ANNOTATION_SIZE = 7.0
PANEL_TITLE_SIZE = 9.0
PANEL_LETTER_SIZE = 10.0

AXIS_LINEWIDTH = 0.8
LINEWIDTH = 1.0
MARKER_SIZE = 24
DEFAULT_DPI = 600


def apply_publication_style():
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.size": BASE_FONT_SIZE,
        "axes.titlesize": PANEL_TITLE_SIZE,
        "axes.labelsize": AXIS_LABEL_SIZE,
        "axes.linewidth": AXIS_LINEWIDTH,
        "xtick.labelsize": TICK_LABEL_SIZE,
        "ytick.labelsize": TICK_LABEL_SIZE,
        "xtick.major.width": AXIS_LINEWIDTH,
        "ytick.major.width": AXIS_LINEWIDTH,
        "legend.fontsize": LEGEND_SIZE,
        "lines.linewidth": LINEWIDTH,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": DEFAULT_DPI,
    })


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=AXIS_LINEWIDTH)
    return ax


def panel_title(ax, letter, title):
    ax.set_title(
        f"{letter}  {title}",
        loc="left",
        fontweight="bold",
        fontsize=PANEL_TITLE_SIZE,
        pad=6,
    )


def save_panel(fig, outdir, basename, dpi=DEFAULT_DPI):
    """Save at a fixed physical size.

    bbox_inches='tight' is deliberately avoided because it changes physical
    panel dimensions and makes 2 x 2 assembly inconsistent.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fig.set_size_inches(*PANEL_SIZE, forward=True)
    fig.savefig(outdir / f"{basename}.png", dpi=dpi)
    fig.savefig(outdir / f"{basename}.pdf")
    plt.close(fig)
