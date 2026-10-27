#!/usr/bin/env python3
"""
Generate Figure 4C-D from virtual-screening result CSVs.

Figure 4C
--------
Positive-control enrichment:
- plots the 4 unseen positive controls by novel rank
- shades the top 40% region
- annotates the exact top-40% enrichment statistics

Figure 4D
--------
Top 10 novel IFRA candidates:
- horizontal bar plot of mean predicted pIC50
- labels each bar with receptors above pIC50 threshold / 10

Inputs can be provided directly, or inferred from --result_dir.
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


def shorten_name(name, max_len=42):
    name = str(name)
    if len(name) <= max_len:
        return name
    return name[: max_len - 3] + "..."


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--result_dir",
        default="results_ifra_indole_aac_gbr",
    )

    p.add_argument(
        "--positive_controls",
        default=None,
        help="positive_control_virtual_screen_ranks.csv",
    )

    p.add_argument(
        "--top_enrichment",
        default=None,
        help="top_percentage_enrichment.csv",
    )

    p.add_argument(
        "--rank_enrichment",
        default=None,
        help="rank_enrichment_summary.csv",
    )

    p.add_argument(
        "--ranking",
        default=None,
        help="ifra_compound_ranking_novel_only.csv",
    )

    p.add_argument(
        "--outdir",
        default="figures/figure4",
    )

    p.add_argument(
        "--top_n",
        type=int,
        default=10,
    )

    p.add_argument(
        "--dpi",
        type=int,
        default=600,
    )

    return p.parse_args()


def save(fig, outdir, basename, dpi):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        outdir / f"{basename}.png",
        dpi=dpi,
        bbox_inches="tight",
    )
    fig.savefig(
        outdir / f"{basename}.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def resolve_paths(args):
    base = Path(args.result_dir)

    pc = (
        Path(args.positive_controls)
        if args.positive_controls
        else base / "positive_control_virtual_screen_ranks.csv"
    )

    top = (
        Path(args.top_enrichment)
        if args.top_enrichment
        else base / "positive_control_enrichment" / "top_percentage_enrichment.csv"
    )

    rank = (
        Path(args.rank_enrichment)
        if args.rank_enrichment
        else base / "positive_control_enrichment" / "rank_enrichment_summary.csv"
    )

    ranking = (
        Path(args.ranking)
        if args.ranking
        else base / "ifra_compound_ranking_novel_only.csv"
    )

    return pc, top, rank, ranking


def plot_figure4c(pc_file, top_file, rank_file, outdir, dpi):
    pc = pd.read_csv(pc_file)
    enrich = pd.read_csv(top_file)
    rank_summary = pd.read_csv(rank_file)

    if "is_training_compound" in pc.columns:
        pc["is_training_compound"] = pc["is_training_compound"].apply(as_bool)
        pc = pc[~pc["is_training_compound"]].copy()

    pc = pc.dropna(
        subset=["novel_broad_spectrum_rank"]
    ).copy()

    pc = pc.sort_values(
        "novel_broad_spectrum_rank"
    ).reset_index(drop=True)

    if len(pc) == 0:
        raise RuntimeError("No unseen positive controls found.")

    N = int(pc["novel_candidate_count"].dropna().iloc[0])

    row40 = enrich.loc[
        enrich["top_percentage"] == 40.0
    ]
    if len(row40) == 0:
        raise RuntimeError("Top-40% enrichment row not found.")

    row40 = row40.iloc[0]

    cutoff = int(row40["rank_cutoff"])
    observed = int(row40["observed_controls_in_top_bin"])
    expected = float(row40["expected_controls_in_top_bin"])
    p_hyper = float(row40["hypergeometric_p_one_sided"])
    p_perm = float(row40["empirical_p_one_sided"])

    mean_row = rank_summary.loc[
        rank_summary["analysis"] == "mean_rank"
    ]
    mean_rank_p = None
    if len(mean_row):
        mean_rank_p = float(
            mean_row.iloc[0]["empirical_p_one_sided"]
        )

    fig, ax = plt.subplots(figsize=PANEL_SIZE)

    # Top 40% region.
    ax.axvspan(
        1,
        cutoff,
        alpha=0.12,
        zorder=0,
    )

    y = list(range(len(pc)))

    ax.scatter(
        pc["novel_broad_spectrum_rank"],
        y,
        s=MARKER_SIZE,
        edgecolors="black",
        linewidths=0.8,
        zorder=3,
    )

    for yi, (_, row) in zip(y, pc.iterrows()):
        rank = int(row["novel_broad_spectrum_rank"])
        ax.text(
            rank + 28,
            yi,
            f"Rank {rank}",
            va="center",
            fontsize=9,
        )

    ax.axvline(
        cutoff,
        linestyle="--",
        linewidth=1.2,
    )

    ax.text(
        cutoff,
        len(pc) - 0.15,
        f"Top 40% cutoff\n(rank {cutoff})",
        ha="right",
        va="bottom",
        fontsize=6.5,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(
        pc["positive_control_name"].astype(str),
        fontsize=7,
    )

    ax.invert_yaxis()

    ax.set_xlim(0, N + 50)
    ax.set_xlabel(
        f"Rank among {N} novel IFRA compounds (lower is better)"
    )

    ax.set_title(
        "C  Enrichment of unseen positive controls",
        loc="left",
        fontweight="bold",
        fontsize=15,
        pad=12,
    )

    stats_text = (
        f"{observed}/{len(pc)} controls within top 40%\n"
        f"Expected = {expected:.2f}\n"
        f"Hypergeometric p = {p_hyper:.4f}\n"
        f"Permutation p = {p_perm:.4f}"
    )

    if mean_rank_p is not None:
        stats_text += f"\nMean-rank permutation p = {mean_rank_p:.4f}"

    ax.text(
        0.98,
        0.96,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.5,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="0.7",
            linewidth=0.8,
        ),
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=0.8)

    save(
        fig,
        outdir,
        "Figure4C_positive_control_enrichment",
        dpi,
    )


def plot_figure4d(ranking_file, outdir, top_n, dpi):
    d = pd.read_csv(ranking_file)

    required = [
        "novel_broad_spectrum_rank",
        "pubchem_title",
        "mean_predicted_pIC50",
        "n_receptors_above_threshold",
    ]

    missing = [c for c in required if c not in d.columns]
    if missing:
        raise ValueError(
            f"Ranking file missing columns: {missing}"
        )

    top = (
        d.sort_values("novel_broad_spectrum_rank")
        .head(top_n)
        .copy()
    )

    top["label"] = [
        f"{int(r)}. {shorten_name(n)}"
        for r, n in zip(
            top["novel_broad_spectrum_rank"],
            top["pubchem_title"],
        )
    ]

    # Reverse so rank 1 appears at top of horizontal plot.
    top = top.iloc[::-1].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=PANEL_SIZE)

    bars = ax.barh(
        top["label"],
        top["mean_predicted_pIC50"],
    )

    xmin = max(
        0,
        float(top["mean_predicted_pIC50"].min()) - 0.10
    )
    xmax = float(top["mean_predicted_pIC50"].max()) + 0.12
    ax.set_xlim(xmin, xmax)

    for bar, score, n_active in zip(
        bars,
        top["mean_predicted_pIC50"],
        top["n_receptors_above_threshold"],
    ):
        ax.text(
            float(score) + 0.006,
            bar.get_y() + bar.get_height() / 2,
            f"{float(score):.3f}   ({int(n_active)}/10)",
            va="center",
            fontsize=6.3,
        )

    ax.set_xlabel(
        "Mean predicted pIC50 across 10 receptors"
    )

    ax.set_title(
        f"D  Top {top_n} predicted broad-spectrum candidates",
        loc="left",
        fontweight="bold",
        fontsize=15,
        pad=12,
    )

    ax.text(
        0.99,
        0.02,
        "(n/10) = receptors with predicted pIC50 ≥ 4",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.5,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.subplots_adjust(left=0.46, right=0.98, top=0.84, bottom=0.18)

    save(
        fig,
        outdir,
        "Figure4D_top_candidates",
        dpi,
    )


def main():
    args = parse_args()
    pc_file, top_file, rank_file, ranking_file = resolve_paths(args)

    for f in [pc_file, top_file, rank_file, ranking_file]:
        if not f.exists():
            raise FileNotFoundError(f)

    print("=== Generate Figure 4C-D ===")
    print("Positive controls:", pc_file)
    print("Top enrichment:", top_file)
    print("Rank enrichment:", rank_file)
    print("Novel ranking:", ranking_file)

    plot_figure4c(
        pc_file,
        top_file,
        rank_file,
        args.outdir,
        args.dpi,
    )

    plot_figure4d(
        ranking_file,
        args.outdir,
        args.top_n,
        args.dpi,
    )

    print()
    print(
        "Outputs written to:",
        Path(args.outdir).resolve(),
    )


if __name__ == "__main__":
    main()
