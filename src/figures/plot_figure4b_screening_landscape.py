#!/usr/bin/env python3
"""
Figure 4B: screening landscape with NO on-plot text annotations.

Positive controls are shown only as highlighted points and identified in a
legend placed outside the axes. This version is intentionally named v2 so that
it cannot be confused with an older local copy.
"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from figure_style import (
    PANEL_SIZE, ANNOTATION_SIZE, LEGEND_SIZE, MARKER_SIZE,
    apply_publication_style, style_axes, panel_title, save_panel,
)

apply_publication_style()


def as_bool(x):
    if isinstance(x, bool):
        return x
    if pd.isna(x):
        return False
    return str(x).strip().lower() in {"true", "1", "yes", "y"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", default="results_ifra_indole_aac_gbr")
    ap.add_argument("--outdir", default="figures/figure4")
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()

    result_dir = Path(args.result_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ranking_file = result_dir / "ifra_compound_ranking_novel_only.csv"
    pc_file = result_dir / "positive_control_virtual_screen_ranks.csv"

    d = pd.read_csv(ranking_file).sort_values("novel_broad_spectrum_rank").copy()
    pc = pd.read_csv(pc_file).copy()

    if "is_training_compound" in pc.columns:
        pc["is_training_compound"] = pc["is_training_compound"].apply(as_bool)
        pc = pc[~pc["is_training_compound"]].copy()

    pc = pc.dropna(
        subset=["novel_broad_spectrum_rank", "mean_predicted_pIC50"]
    ).sort_values("novel_broad_spectrum_rank")

    fig, ax = plt.subplots(figsize=PANEL_SIZE)

    ax.scatter(
        d["novel_broad_spectrum_rank"],
        d["mean_predicted_pIC50"],
        s=10,
        alpha=0.38,
        linewidths=0,
        label="Novel IFRA compounds",
        zorder=1,
    )

    for _, row in pc.iterrows():
        rank = int(row["novel_broad_spectrum_rank"])
        score = float(row["mean_predicted_pIC50"])
        name = str(row.get("positive_control_name", "Control"))

        ax.scatter(
            rank,
            score,
            s=MARKER_SIZE,
            edgecolors="black",
            linewidths=0.8,
            label=f"{name} (rank {rank})",
            zorder=4,
        )

    ax.set_xlabel("Rank among novel IFRA compounds")
    ax.set_ylabel("Mean predicted pIC50 across 10 receptors")
    ax.set_title(
        "B  Virtual-screening landscape",
        loc="left",
        fontweight="bold",
        fontsize=15,
        pad=12,
    )

    ax.legend(
        loc="upper right",
        frameon=False,
        fontsize=7,
        borderaxespad=0.0,
        labelspacing=0.8,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.20, top=0.84)

    save_panel(fig, outdir, "Figure4B_screening_landscape", args.dpi)

    print("SCRIPT VERSION: Figure4B legend-only")
    print("NO ax.annotate() calls are used.")
    print("Outputs written to:", outdir.resolve())


if __name__ == "__main__":
    main()
