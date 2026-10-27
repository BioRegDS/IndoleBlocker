#!/usr/bin/env python3
"""
Generate Figure 1B-C from the prepared Indole IC50 dataset.

B: receptor-wise distribution of measured pIC50 values (37 records)
C: blocker x receptor pIC50 heatmap with missing combinations shown blank

Figure 1A is the manually assembled workflow schematic and is not generated here.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from figure_style import (
    PANEL_SIZE, ANNOTATION_SIZE, LEGEND_SIZE, MARKER_SIZE,
    apply_publication_style, style_axes, panel_title, save_panel,
)

apply_publication_style()



def short_name(s, n=34):
    s = str(s)
    return s if len(s) <= n else s[: n - 3] + "..."


def save(fig, outdir, name, dpi):
    save_panel(fig, outdir, name, dpi)


def panel_b(d, outdir, dpi):
    receptors = sorted(d["receptor"].unique())
    fig, ax = plt.subplots(figsize=PANEL_SIZE)

    rng = np.random.default_rng(42)
    for i, r in enumerate(receptors):
        y = d.loc[d["receptor"] == r, "pActivity_M_recalculated"].to_numpy(float)
        x = i + rng.normal(0, 0.055, size=len(y))
        ax.scatter(x, y, s=MARKER_SIZE, alpha=0.8)
        if len(y):
            med = float(np.median(y))
            ax.plot([i - 0.18, i + 0.18], [med, med], linewidth=2)

    ax.set_xticks(range(len(receptors)))
    ax.set_xticklabels(receptors, rotation=45, ha="right")
    ax.set_ylabel("Measured pIC50")
    ax.set_xlabel("Indole-responsive receptor")
    ax.set_title("B  Distribution of measured Indole antagonist activity", loc="left", fontweight="bold", fontsize=9, pad=6)
    ax.text(0.98, 0.96, f"n = {len(d)} measurements", transform=ax.transAxes, ha="right", va="top", fontsize=6.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(pad=0.8)
    save(fig, outdir, "Figure1B_indole_pIC50_distribution", dpi)


def panel_c(d, outdir, dpi):
    receptor_order = [
        "Olfr738", "Olfr739", "Olfr740", "Olfr741", "Olfr742",
        "Olfr743", "Olfr744", "Olfr746", "Olfr748", "Olfr749",
    ]
    receptors = [r for r in receptor_order if r in set(d["receptor"])]

    # Order blockers by mean pIC50 for readability.
    blocker_order = (
        d.groupby("blocker")["pActivity_M_recalculated"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

    mat = d.pivot_table(
        index="blocker",
        columns="receptor",
        values="pActivity_M_recalculated",
        aggfunc="mean",
    ).reindex(index=blocker_order, columns=receptors)

    arr = mat.to_numpy(float)
    masked = np.ma.masked_invalid(arr)

    fig, ax = plt.subplots(figsize=PANEL_SIZE)
    im = ax.imshow(masked, aspect="auto", interpolation="nearest")

    ax.set_xticks(range(len(receptors)))
    ax.set_xticklabels(receptors, rotation=45, ha="right")
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels([short_name(x) for x in mat.index], fontsize=6.5)
    ax.set_xlabel("Receptor")
    ax.set_ylabel("Antagonist")
    ax.set_title("C  Indole antagonist-receptor activity matrix", loc="left", fontweight="bold", fontsize=9, pad=6)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("Measured pIC50", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # Put values only where measurements exist.
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if np.isfinite(arr[i, j]):
                ax.text(j, i, f"{arr[i,j]:.2f}", ha="center", va="center", fontsize=6.2)

    fig.tight_layout(pad=0.8)
    save(fig, outdir, "Figure1C_indole_activity_heatmap", dpi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/dataset/indole_ic50_dataset.csv")
    ap.add_argument("--outdir", default="figures/figure1")
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()

    d = pd.read_csv(args.input)
    required = ["receptor", "blocker", "pActivity_M_recalculated"]
    missing = [c for c in required if c not in d.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    print("=== Generate Figure 1B-C ===")
    print(f"Records: {len(d)}")
    print(f"Blockers: {d['blocker'].nunique()}")
    print(f"Receptors: {d['receptor'].nunique()}")

    panel_b(d, args.outdir, args.dpi)
    panel_c(d, args.outdir, args.dpi)

    print(f"Outputs written to: {Path(args.outdir).resolve()}")


if __name__ == "__main__":
    main()
