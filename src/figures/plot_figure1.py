#!/usr/bin/env python3
"""
Generate composite Figure 1.

Layout
------
A: upper panel spanning full width, intentionally blank for manual insertion
   of the Study workflow.
B: lower-left, receptor-wise distribution of measured pIC50.
C: lower-right, antagonist x receptor pIC50 heatmap.
Bottom: 3 x 3 antagonist key (Comp1-Comp9).

Full compound names remain available in Supplementary Table S1.
"""

import argparse
from pathlib import Path
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Two-column figure width (~175 mm), with enough height for A + B/C + key.
FIGSIZE = (6.9, 6.3)


def save(fig, outdir, name, dpi):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{name}.png", dpi=dpi)
    fig.savefig(outdir / f"{name}.pdf")
    plt.close(fig)


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def panel_title(ax, letter, title):
    ax.set_title(
        f"{letter}  {title}",
        loc="left",
        fontweight="bold",
        fontsize=9,
        pad=5,
    )


def wrap_full_name(s, width=28):
    """Wrap the complete compound name over multiple lines without truncation."""
    s = str(s)
    return "\n".join(textwrap.wrap(
        s,
        width=width,
        break_long_words=False,
        break_on_hyphens=True,
    ))


def prepare_heatmap(d):
    receptor_order = [
        "Olfr738", "Olfr739", "Olfr740", "Olfr741", "Olfr742",
        "Olfr743", "Olfr744", "Olfr746", "Olfr748", "Olfr749",
    ]
    receptors = [r for r in receptor_order if r in set(d["receptor"])]

    blocker_order = (
        d.groupby("blocker")["pActivity_M_recalculated"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

    mat = (
        d.pivot_table(
            index="blocker",
            columns="receptor",
            values="pActivity_M_recalculated",
            aggfunc="mean",
        )
        .reindex(index=blocker_order, columns=receptors)
    )

    blocker_ids = {blocker: f"Comp{i+1}" for i, blocker in enumerate(blocker_order)}
    return mat, receptors, blocker_order, blocker_ids


def draw_panel_b(ax, d):
    receptor_order = [
        "Olfr738", "Olfr739", "Olfr740", "Olfr741", "Olfr742",
        "Olfr743", "Olfr744", "Olfr746", "Olfr748", "Olfr749",
    ]
    receptors = [r for r in receptor_order if r in set(d["receptor"])]

    rng = np.random.default_rng(42)

    for i, receptor in enumerate(receptors):
        y = (
            d.loc[d["receptor"] == receptor, "pActivity_M_recalculated"]
            .to_numpy(float)
        )
        x = i + rng.normal(0, 0.055, size=len(y))

        ax.scatter(x, y, s=18, alpha=0.80, zorder=2)

        if len(y):
            median = float(np.median(y))
            ax.plot(
                [i - 0.18, i + 0.18],
                [median, median],
                linewidth=1.4,
                zorder=3,
            )

    ax.set_xticks(range(len(receptors)))
    ax.set_xticklabels(
        receptors,
        rotation=55,
        ha="right",
        rotation_mode="anchor",
        fontsize=6.2,
    )
    ax.set_ylabel("Measured pIC50")
    ax.set_xlabel("Indole-responsive receptor")
    panel_title(ax, "B", "Measured antagonist activity")

    ax.text(
        0.98, 0.96,
        f"n = {len(d)}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.5,
    )
    style_axes(ax)


def draw_panel_c(ax, mat, receptors, blocker_ids):
    arr = mat.to_numpy(float)
    masked = np.ma.masked_invalid(arr)

    im = ax.imshow(
        masked,
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_xticks(range(len(receptors)))
    ax.set_xticklabels(
        receptors,
        rotation=55,
        ha="right",
        rotation_mode="anchor",
        fontsize=5.8,
    )

    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(
        [blocker_ids[x] for x in mat.index],
        fontsize=6.5,
    )

    ax.set_xlabel("Receptor")
    ax.set_ylabel("Antagonist")
    panel_title(ax, "C", "Antagonist–receptor activity")

    cbar = ax.figure.colorbar(
        im,
        ax=ax,
        fraction=0.045,
        pad=0.025,
    )
    cbar.set_label(
        "Measured pIC50",
        fontsize=6.5,
        labelpad=4,
    )
    cbar.ax.tick_params(
        labelsize=5.8,
        width=0.7,
    )


def draw_antagonist_key(ax, blocker_order, blocker_ids):
    """Draw a two-column compound key with complete, wrapped names."""
    ax.set_axis_off()

    ax.text(
        0.0, 0.88,
        "Antagonist key",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
        fontweight="bold",
    )

    ax.text(
        1.0, 0.88,
        "Source data: Supplementary Table S1",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.5,
        style="italic",
    )

    # Two columns are used because several compound names are too long
    # for a readable 3-column layout at publication width.
    x_positions = [0.00, 0.52]
    y_positions = [0.66, 0.545, 0.430, 0.315, 0.200]

    for i, blocker in enumerate(blocker_order):
        if i < 5:
            col = 0
            row = i
        else:
            col = 1
            row = i - 5

        # Wrap the complete name without truncation.
        wrapped = "\n".join(textwrap.wrap(
            str(blocker),
            width=38,
            break_long_words=False,
            break_on_hyphens=True,
        ))

        ax.text(
            x_positions[col],
            y_positions[row],
            f"{blocker_ids[blocker]}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=5.8,
            fontweight="bold",
        )

        ax.text(
            x_positions[col] + 0.065,
            y_positions[row],
            wrapped,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=5.4,
            linespacing=0.85,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="results/dataset/indole_ic50_dataset.csv",
    )
    ap.add_argument(
        "--outdir",
        default="figures/figure1",
    )
    ap.add_argument(
        "--dpi",
        type=int,
        default=600,
    )
    args = ap.parse_args()

    d = pd.read_csv(args.input)

    required = [
        "receptor",
        "blocker",
        "pActivity_M_recalculated",
    ]
    missing = [c for c in required if c not in d.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    print("=== Generate composite Figure 1 ===")
    print(f"Records: {len(d)}")
    print(f"Blockers: {d['blocker'].nunique()}")
    print(f"Receptors: {d['receptor'].nunique()}")

    mat, receptors, blocker_order, blocker_ids = prepare_heatmap(d)

    fig = plt.figure(figsize=FIGSIZE)

    # A = 35%, B/C = 50%, key = 15%.
    gs = fig.add_gridspec(
        nrows=3,
        ncols=2,
        height_ratios=[0.36, 0.36, 0.28],
        width_ratios=[1.0, 1.0],
        hspace=0.34,
        wspace=0.38,
    )

    # A: reserved blank area for Study workflow.
    ax_a = fig.add_subplot(gs[0, :])
    ax_a.set_axis_off()
    ax_a.text(
        0.0, 1.0, "A  Study workflow",
        transform=ax_a.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
    )

    # B / C.
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    draw_panel_b(ax_b, d)
    draw_panel_c(ax_c, mat, receptors, blocker_ids)

    # Full-width antagonist key.
    ax_key = fig.add_subplot(gs[2, :])
    draw_antagonist_key(ax_key, blocker_order, blocker_ids)

    fig.subplots_adjust(
        left=0.10,
        right=0.97,
        bottom=0.035,
        top=0.97,
    )

    save(
        fig,
        args.outdir,
        "Figure1_composite",
        args.dpi,
    )

    print("Antagonist IDs:")
    for blocker in blocker_order:
        print(f"  {blocker_ids[blocker]}: {blocker}")

    print("Outputs written to:", Path(args.outdir).resolve())


if __name__ == "__main__":
    main()
