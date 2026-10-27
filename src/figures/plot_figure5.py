#!/usr/bin/env python3
"""
Generate composite Figure 5 from the manually curated Top-50
external-evidence annotation table.

Layout
------
A: upper-left, external-evidence grade across virtual-screen ranks 1-50.
B: upper-right, distribution of evidence grades.
C: lower full-width panel, representative literature/patent-supported hits.

Evidence grades
---------------
A = direct olfactory-receptor antagonist / odor-blocking evidence
B = sensory malodor-suppression evidence
C = malodor-counteraction patent evidence
D = related chemosensory activity
E = no qualifying evidence identified in the targeted search

No external data are added here; the plot is generated only from the input CSV.
"""

import argparse
from pathlib import Path
import textwrap

import numpy as np
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

FIGSIZE = (6.9, 5.25)

GRADE_ORDER = ["A", "B", "C", "D", "E"]
GRADE_Y = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1}
GRADE_LABELS = {
    "A": "Direct OR antagonist",
    "B": "Sensory malodor suppression",
    "C": "Malodor-counteraction patent",
    "D": "Related chemosensory activity",
    "E": "No qualifying evidence",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--outdir", default="figures/figure5")
    p.add_argument("--dpi", type=int, default=600)
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


def choose_representatives(df):
    reps = []
    for grade in ["A", "B", "C"]:
        z = df[df["grade"] == grade].sort_values("novel_rank")
        if len(z):
            reps.append(z.iloc[0])
    return reps


# ---------------------------------------------------------------------
# Panel A
# ---------------------------------------------------------------------
def draw_panel_a(ax, df):
    for g in GRADE_ORDER:
        ax.axhline(
            GRADE_Y[g],
            linewidth=0.6,
            alpha=0.22,
            zorder=0,
        )

    # Plot each grade separately so grade groups remain visually distinct.
    for g in GRADE_ORDER:
        z = df[df["grade"] == g]
        ax.scatter(
            z["novel_rank"],
            [GRADE_Y[g]] * len(z),
            s=22,
            zorder=3,
        )

    ax.set_xlim(0, 51)
    ax.set_ylim(0.5, 5.5)
    ax.set_xticks([1, 10, 20, 30, 40, 50])
    ax.set_yticks([5, 4, 3, 2, 1])
    ax.set_yticklabels([
        "A  Direct OR antagonist",
        "B  Sensory suppression",
        "C  Malodor patent",
        "D  Related activity",
        "E  No qualifying evidence",
    ], fontsize=5.8)

    ax.set_xlabel("Virtual-screen rank")
    ax.set_ylabel("External evidence grade")

    panel_title(
        ax,
        "A",
        "External evidence across Top 50",
    )

    style_axes(ax)


# ---------------------------------------------------------------------
# Panel B
# ---------------------------------------------------------------------
def draw_panel_b(ax, df):
    counts = (
        df["grade"]
        .value_counts()
        .reindex(GRADE_ORDER, fill_value=0)
    )

    n_total = len(df)
    n_abc = int(
        counts["A"]
        + counts["B"]
        + counts["C"]
    )
    n_abcd = int(
        n_abc
        + counts["D"]
    )

    bars = ax.bar(
        GRADE_ORDER,
        counts.values,
        width=0.68,
    )

    for bar, n in zip(
        bars,
        counts.values,
    ):
        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            n + 0.6,
            str(int(n)),
            ha="center",
            va="bottom",
            fontsize=6.2,
            fontweight="bold",
        )

    ax.set_ylabel("Number of compounds")
    ax.set_xlabel("External evidence grade")
    ax.set_ylim(
        0,
        max(counts.values) * 1.24,
    )

    panel_title(
        ax,
        "B",
        "Evidence grade distribution",
    )

    ax.text(
        0.97,
        0.95,
        (
            f"A–C: {n_abc}/{n_total} ({100*n_abc/n_total:.0f}%)\n"
            f"A–D: {n_abcd}/{n_total} ({100*n_abcd/n_total:.0f}%)"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.0,
        bbox=dict(
            boxstyle="round,pad=0.18",
            facecolor="white",
            edgecolor="none",
            alpha=0.88,
        ),
    )

    style_axes(ax)


# ---------------------------------------------------------------------
# Panel C
# ---------------------------------------------------------------------
def draw_panel_c(ax, df):
    reps = choose_representatives(df)

    ax.set_axis_off()

    ax.text(
        0.0,
        1.0,
        "C  Representative externally supported candidates",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
    )

    # Column anchors.
    x_name = 0.02
    x_pred = 0.29
    x_grade = 0.47
    x_evidence = 0.56

    # Header.
    ax.text(
        x_name, 0.84, "Candidate",
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=6.5,
    )
    ax.text(
        x_pred, 0.84, "Prediction",
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=6.5,
    )
    ax.text(
        x_grade, 0.84, "Grade",
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=6.5,
        ha="center",
    )
    ax.text(
        x_evidence, 0.84, "External evidence",
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=6.5,
    )

    y_positions = [0.64, 0.40, 0.16]

    for idx, (y, row) in enumerate(
        zip(y_positions, reps)
    ):
        name = str(row["pubchem_title"])
        rank = int(row["novel_rank"])
        meanp = float(
            row["mean_predicted_pIC50"]
        )
        minp = float(
            row["min_predicted_pIC50"]
        )
        grade = str(row["grade"])

        # Keep candidate names compact but readable.
        name_wrapped = "\n".join(
            textwrap.wrap(
                name,
                width=24,
                break_long_words=False,
                break_on_hyphens=True,
            )
        )

        ev = str(row["evidence_summary"])
        ev_wrapped = "\n".join(
            textwrap.wrap(
                ev,
                width=48,
                break_long_words=False,
                break_on_hyphens=True,
            )
        )

        ax.text(
            x_name,
            y,
            f"{name_wrapped}\nRank {rank}",
            transform=ax.transAxes,
            va="center",
            fontsize=5.6,
        )

        ax.text(
            x_pred,
            y,
            (
                f"Mean pIC50 = {meanp:.3f}\n"
                f"Min pIC50 = {minp:.3f}"
            ),
            transform=ax.transAxes,
            va="center",
            fontsize=5.6,
        )

        ax.text(
            x_grade,
            y,
            grade,
            transform=ax.transAxes,
            va="center",
            ha="center",
            fontsize=8.0,
            fontweight="bold",
            bbox=dict(
                boxstyle="circle,pad=0.38",
                facecolor="white",
                edgecolor="0.4",
                linewidth=0.8,
            ),
        )

        ax.text(
            x_evidence,
            y,
            ev_wrapped,
            transform=ax.transAxes,
            va="center",
            fontsize=5.2,
        )

        if idx < len(reps) - 1:
            ax.plot(
                [0.02, 0.98],
                [y - 0.12, y - 0.12],
                transform=ax.transAxes,
                linewidth=0.6,
                alpha=0.30,
            )

    ax.text(
        0.02,
        0.01,
        "Representative hits are the highest-ranked examples of grades A, B, and C.",
        transform=ax.transAxes,
        fontsize=5.4,
        style="italic",
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    args = parse_args()

    df = pd.read_csv(args.input)

    required = [
        "novel_rank",
        "pubchem_title",
        "mean_predicted_pIC50",
        "min_predicted_pIC50",
        "grade",
        "evidence_summary",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    df = df.copy()
    df["grade"] = (
        df["grade"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df = (
        df[
            df["grade"].isin(
                GRADE_ORDER
            )
        ]
        .sort_values(
            "novel_rank"
        )
        .reset_index(drop=True)
    )

    print("=== Generate composite Figure 5 ===")
    print("Rows:", len(df))
    print("Grade counts:")
    print(
        df["grade"]
        .value_counts()
        .reindex(
            GRADE_ORDER,
            fill_value=0,
        )
        .to_string()
    )

    fig = plt.figure(
        figsize=FIGSIZE,
    )

    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        height_ratios=[0.48, 0.52],
        width_ratios=[1.0, 1.0],
        hspace=0.42,
        wspace=0.38,
    )

    ax_a = fig.add_subplot(
        gs[0, 0]
    )
    ax_b = fig.add_subplot(
        gs[0, 1]
    )
    ax_c = fig.add_subplot(
        gs[1, :]
    )

    draw_panel_a(
        ax_a,
        df,
    )
    draw_panel_b(
        ax_b,
        df,
    )
    draw_panel_c(
        ax_c,
        df,
    )

    # Leave enough room for the relatively long y-axis tick labels in panel A.
    # The previous left=0.11 clipped the beginning of labels such as
    # "A  Direct OR antagonist" in the exported figure.
    fig.subplots_adjust(
        left=0.18,
        right=0.98,
        bottom=0.07,
        top=0.96,
    )

    save(
        fig,
        args.outdir,
        "Figure5_composite",
        args.dpi,
    )

    print("\nRepresentative candidates:")
    for r in choose_representatives(df):
        print(
            f"  Grade {r['grade']}: "
            f"rank {int(r['novel_rank'])}, "
            f"{r['pubchem_title']}"
        )

    print(
        "\nOutputs written to:",
        Path(args.outdir).resolve(),
    )


if __name__ == "__main__":
    main()
