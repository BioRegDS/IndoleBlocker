#!/usr/bin/env python3
"""
Generate composite Figure 4.

Layout
------
A: upper-left, intentionally left blank for manual insertion of the
   IFRA/PubChem virtual-screening workflow.
B: upper-right, virtual-screening landscape.
C: lower-left, unseen positive-control enrichment.
D: lower-right, top-ranked broad-spectrum candidates.

Designed to match the compact two-column layout used for Figures 2 and 3.
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Publication style
# ---------------------------------------------------------------------
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

FIGSIZE = (7.2, 5.8)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--result_dir",
        default="results/virtual_screening/predictions",
    )
    p.add_argument(
        "--positive_controls",
        default=None,
    )
    p.add_argument(
        "--top_enrichment",
        default=None,
    )
    p.add_argument(
        "--rank_enrichment",
        default=None,
    )
    p.add_argument(
        "--ranking",
        default=None,
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


def as_bool(x):
    if isinstance(x, bool):
        return x
    if pd.isna(x):
        return False
    return str(x).strip().lower() in {
        "true", "1", "yes", "y"
    }


def shorten_name(name, max_len=28):
    name = str(name)
    if len(name) <= max_len:
        return name
    return name[: max_len - 3] + "..."


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
        else base.parent
        / "positive_control_enrichment"
        / "top_percentage_enrichment.csv"
    )

    rank = (
        Path(args.rank_enrichment)
        if args.rank_enrichment
        else base.parent
        / "positive_control_enrichment"
        / "rank_enrichment_summary.csv"
    )

    ranking = (
        Path(args.ranking)
        if args.ranking
        else base / "ifra_compound_ranking_novel_only.csv"
    )

    return pc, top, rank, ranking


# ---------------------------------------------------------------------
# Panel B
# ---------------------------------------------------------------------
def draw_panel_b(ax, ranking_file, pc_file):
    d = (
        pd.read_csv(ranking_file)
        .sort_values("novel_broad_spectrum_rank")
        .copy()
    )

    pc = pd.read_csv(pc_file).copy()

    if "is_training_compound" in pc.columns:
        pc["is_training_compound"] = (
            pc["is_training_compound"]
            .apply(as_bool)
        )
        pc = pc[
            ~pc["is_training_compound"]
        ].copy()

    pc = (
        pc.dropna(
            subset=[
                "novel_broad_spectrum_rank",
                "mean_predicted_pIC50",
            ]
        )
        .sort_values(
            "novel_broad_spectrum_rank"
        )
    )

    ax.scatter(
        d["novel_broad_spectrum_rank"],
        d["mean_predicted_pIC50"],
        s=4,
        alpha=0.30,
        linewidths=0,
        label="Novel IFRA compounds",
        zorder=1,
    )

    for _, row in pc.iterrows():
        rank = int(
            row["novel_broad_spectrum_rank"]
        )
        score = float(
            row["mean_predicted_pIC50"]
        )
        name = str(
            row.get(
                "positive_control_name",
                "Control",
            )
        )

        ax.scatter(
            rank,
            score,
            s=24,
            edgecolors="black",
            linewidths=0.6,
            label=f"{name} ({rank})",
            zorder=4,
        )

    ax.set_xlabel(
        "Rank among novel IFRA compounds"
    )
    ax.set_ylabel(
        "Mean predicted pIC50"
    )

    panel_title(
        ax,
        "B",
        "Virtual-screening landscape",
    )

    ax.legend(
        loc="upper right",
        frameon=False,
        fontsize=5.6,
        labelspacing=0.35,
        handletextpad=0.4,
        borderaxespad=0.2,
    )

    style_axes(ax)


# ---------------------------------------------------------------------
# Panel C
# ---------------------------------------------------------------------
def draw_panel_c(
    ax,
    pc_file,
    top_file,
    rank_file,
):
    pc = pd.read_csv(pc_file)
    enrich = pd.read_csv(top_file)
    rank_summary = pd.read_csv(rank_file)

    if "is_training_compound" in pc.columns:
        pc["is_training_compound"] = (
            pc["is_training_compound"]
            .apply(as_bool)
        )
        pc = pc[
            ~pc["is_training_compound"]
        ].copy()

    pc = (
        pc.dropna(
            subset=["novel_broad_spectrum_rank"]
        )
        .sort_values(
            "novel_broad_spectrum_rank"
        )
        .reset_index(drop=True)
    )

    if len(pc) == 0:
        raise RuntimeError(
            "No unseen positive controls found."
        )

    N = int(
        pc["novel_candidate_count"]
        .dropna()
        .iloc[0]
    )

    row40 = enrich.loc[
        enrich["top_percentage"] == 40.0
    ]

    if len(row40) == 0:
        raise RuntimeError(
            "Top-40% enrichment row not found."
        )

    row40 = row40.iloc[0]

    cutoff = int(
        row40["rank_cutoff"]
    )
    observed = int(
        row40[
            "observed_controls_in_top_bin"
        ]
    )
    expected = float(
        row40[
            "expected_controls_in_top_bin"
        ]
    )
    p_hyper = float(
        row40[
            "hypergeometric_p_one_sided"
        ]
    )
    p_perm = float(
        row40[
            "empirical_p_one_sided"
        ]
    )

    mean_row = rank_summary.loc[
        rank_summary["analysis"] == "mean_rank"
    ]

    mean_rank_p = None
    if len(mean_row):
        mean_rank_p = float(
            mean_row.iloc[0][
                "empirical_p_one_sided"
            ]
        )

    ax.axvspan(
        1,
        cutoff,
        alpha=0.10,
        zorder=0,
    )

    y = list(range(len(pc)))

    ax.scatter(
        pc["novel_broad_spectrum_rank"],
        y,
        s=28,
        edgecolors="black",
        linewidths=0.6,
        zorder=3,
    )

    ax.axvline(
        cutoff,
        linestyle="--",
        linewidth=0.9,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(
        pc["positive_control_name"].astype(str),
        fontsize=5.9,
    )
    ax.invert_yaxis()

    ax.set_xlim(
        0,
        N + 60,
    )

    ax.set_xlabel(
        "Rank among novel IFRA compounds"
    )

    panel_title(
        ax,
        "C",
        "Positive-control enrichment",
    )

    stats_text = (
        f"{observed}/{len(pc)} within top 40%\n"
        f"Expected = {expected:.2f}\n"
        f"Permutation p = {p_perm:.4f}\n"
        f"Hypergeometric p = {p_hyper:.4f}"
    )

    ax.text(
        0.66,
        0.12,
        stats_text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.7,
        bbox=dict(
            boxstyle="round,pad=0.18",
            facecolor="white",
            edgecolor="none",
            alpha=0.88,
        ),
    )

    style_axes(ax)


# ---------------------------------------------------------------------
# Panel D
# ---------------------------------------------------------------------
def draw_panel_d(
    ax,
    ranking_file,
    top_n,
):
    d = pd.read_csv(
        ranking_file
    )

    required = [
        "novel_broad_spectrum_rank",
        "pubchem_title",
        "mean_predicted_pIC50",
        "n_receptors_above_threshold",
    ]

    missing = [
        c for c in required
        if c not in d.columns
    ]

    if missing:
        raise ValueError(
            f"Ranking file missing columns: {missing}"
        )

    top = (
        d.sort_values(
            "novel_broad_spectrum_rank"
        )
        .head(top_n)
        .copy()
    )

    top["compound_id"] = [
        f"Comp{i+1}"
        for i in range(len(top))
    ]

    # Reverse so rank 1 is at the top of the horizontal plot.
    plot_top = (
        top.iloc[::-1]
        .reset_index(drop=True)
    )

    bars = ax.barh(
        plot_top["compound_id"],
        plot_top["mean_predicted_pIC50"],
    )

    xmin = max(
        0,
        float(
            plot_top["mean_predicted_pIC50"].min()
        ) - 0.10,
    )

    xmax = float(
        plot_top["mean_predicted_pIC50"].max()
    ) + 0.10

    ax.set_xlim(
        xmin,
        xmax + 0.015,
    )

    for bar, score, n_active in zip(
        bars,
        plot_top["mean_predicted_pIC50"],
        plot_top["n_receptors_above_threshold"],
    ):
        ax.text(
            float(score) + 0.004,
            bar.get_y()
            + bar.get_height() / 2,
            f"{float(score):.3f} ({int(n_active)}/10)",
            va="center",
            fontsize=5.0,
        )

    ax.set_xlabel(
        "Mean predicted pIC50"
    )

    panel_title(
        ax,
        "D",
        f"Top {top_n} candidates",
    )

    ax.tick_params(
        axis="y",
        labelsize=5.5,
    )

    style_axes(ax)

    return top


def wrap_candidate_name(name, width=34):
    import textwrap
    return "\n".join(
        textwrap.wrap(
            str(name),
            width=width,
            break_long_words=False,
            break_on_hyphens=True,
        )
    )


def draw_candidate_key(ax, top):
    """Draw a compact two-column key below panel D with one-line names."""
    ax.set_axis_off()

    ax.text(
        0.0,
        0.96,
        "Candidate key",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.0,
        fontweight="bold",
    )

    # Two wide columns. All names are shown on one line.
    x_id = [0.00, 0.56]
    x_name = [0.09, 0.65]
    y_positions = [0.74, 0.58, 0.42, 0.26, 0.10]

    for i, (_, row) in enumerate(top.iterrows()):
        if i < 5:
            col = 0
            row_idx = i
        else:
            col = 1
            row_idx = i - 5

        ax.text(
            x_id[col],
            y_positions[row_idx],
            f"{row['compound_id']}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=5.2,
            fontweight="bold",
        )

        ax.text(
            x_name[col],
            y_positions[row_idx],
            str(row["pubchem_title"]),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=4.8,
        )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    args = parse_args()

    pc_file, top_file, rank_file, ranking_file = (
        resolve_paths(args)
    )

    for f in [
        pc_file,
        top_file,
        rank_file,
        ranking_file,
    ]:
        if not f.exists():
            raise FileNotFoundError(f)

    print(
        "=== Generate composite Figure 4 ==="
    )
    print("Positive controls:", pc_file)
    print("Top enrichment:", top_file)
    print("Rank enrichment:", rank_file)
    print("Novel ranking:", ranking_file)

    fig = plt.figure(
        figsize=FIGSIZE,
    )

    gs = fig.add_gridspec(
        nrows=3,
        ncols=2,
        height_ratios=[0.48, 0.37, 0.15],
        width_ratios=[0.90, 1.10],
        hspace=0.45,
        wspace=0.34,
    )

    # A: blank panel for manually assembled workflow schematic.
    ax_a = fig.add_subplot(
        gs[0, 0]
    )
    ax_a.set_axis_off()

    ax_a.text(
        0.0,
        1.025,
        "A  IFRA/PubChem virtual-screening workflow",
        transform=ax_a.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        clip_on=False,
    )

    ax_b = fig.add_subplot(
        gs[0, 1]
    )
    ax_c = fig.add_subplot(
        gs[1, 0]
    )
    ax_d = fig.add_subplot(
        gs[1, 1]
    )

    # Candidate key is placed only below panel D.
    ax_key = fig.add_subplot(
        gs[2, 1]
    )

    # Keep the lower-left area blank.
    ax_blank = fig.add_subplot(
        gs[2, 0]
    )
    ax_blank.set_axis_off()

    draw_panel_b(
        ax_b,
        ranking_file,
        pc_file,
    )

    draw_panel_c(
        ax_c,
        pc_file,
        top_file,
        rank_file,
    )

    top_candidates = draw_panel_d(
        ax_d,
        ranking_file,
        args.top_n,
    )

    draw_candidate_key(
        ax_key,
        top_candidates,
    )

    fig.subplots_adjust(
        left=0.10,
        right=0.985,
        bottom=0.055,
        top=0.965,
    )

    save(
        fig,
        args.outdir,
        "Figure4_composite",
        args.dpi,
    )

    print(
        "Outputs written to:",
        Path(args.outdir).resolve(),
    )


if __name__ == "__main__":
    main()
