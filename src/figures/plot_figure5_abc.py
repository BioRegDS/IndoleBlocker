#!/usr/bin/env python3
"""
Generate Figure 5A-C from Top-50 external-evidence annotation CSV.

A. Evidence grade across virtual-screen ranks 1-50
B. Evidence grade distribution
C. Representative literature/patent-supported candidates

Grade definitions used here:
A = direct olfactory-receptor antagonist evidence
B = sensory malodor-suppression evidence
C = malodor-related patent evidence
D = related chemosensory activity
E = no qualifying evidence identified in the targeted search

No external web data are added; all labels/evidence are taken from the input CSV.
"""

import argparse
from pathlib import Path
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from figure_style import (
    PANEL_SIZE, ANNOTATION_SIZE, LEGEND_SIZE, MARKER_SIZE,
    apply_publication_style, style_axes, panel_title, save_panel,
)

apply_publication_style()


GRADE_ORDER = ["A", "B", "C", "D", "E"]
GRADE_Y = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1}
GRADE_LABELS = {
    "A": "Direct OR antagonist",
    "B": "Sensory malodor suppression",
    "C": "Malodor-related patent",
    "D": "Related chemosensory activity",
    "E": "No qualifying evidence identified",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--outdir", default="figures/figure5")
    p.add_argument("--dpi", type=int, default=600)
    return p.parse_args()


def save(fig, outdir, name, dpi):
    save_panel(fig, outdir, name, dpi)


def shorten(s, width=44):
    s = str(s)
    return textwrap.shorten(s, width=width, placeholder="...")


def panel_a(df, outdir, dpi):
    fig, ax = plt.subplots(figsize=PANEL_SIZE)

    # Draw grade guide lines.
    for g in GRADE_ORDER:
        ax.axhline(GRADE_Y[g], linewidth=0.7, alpha=0.25, zorder=0)

    # Plot each grade separately so the legend is clear.
    for g in GRADE_ORDER:
        z = df[df["grade"] == g]
        ax.scatter(
            z["novel_rank"],
            [GRADE_Y[g]] * len(z),
            s=MARKER_SIZE,
            label=f"{g}: {GRADE_LABELS[g]}",
            zorder=3,
        )

    ax.set_xlim(0, 51)
    ax.set_ylim(0.5, 5.5)
    ax.set_xticks([1, 10, 20, 30, 40, 50])
    ax.set_yticks([5, 4, 3, 2, 1])
    ax.set_yticklabels([
        "A\nDirect OR antagonist",
        "B\nSensory suppression",
        "C\nMalodor patent",
        "D\nRelated activity",
        "E\nNo evidence identified",
    ])
    ax.set_xlabel("Virtual-screen rank among novel IFRA compounds")
    ax.set_ylabel("External evidence grade")
    ax.set_title(
        "A  External evidence across the Top 50 virtual-screen hits",
        loc="left", fontweight="bold", fontsize=9, pad=6
    )

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        fontsize=6.5,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.34, right=0.98, bottom=0.20, top=0.84)
    save(fig, outdir, "Figure5A_top50_evidence_map", dpi)


def panel_b(df, outdir, dpi):
    counts = df["grade"].value_counts().reindex(GRADE_ORDER, fill_value=0)
    n_total = len(df)
    n_supported = int(counts["A"] + counts["B"] + counts["C"] + counts["D"])
    n_direct_or_malodor = int(counts["A"] + counts["B"] + counts["C"])

    fig, ax = plt.subplots(figsize=PANEL_SIZE)
    bars = ax.bar(GRADE_ORDER, counts.values, width=0.68)

    for bar, n in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            n + 0.6,
            str(int(n)),
            ha="center", va="bottom", fontsize=10
        )

    ax.set_ylabel("Number of compounds")
    ax.set_xlabel("External evidence grade")
    ax.set_title(
        "B  Evidence grade distribution among Top 50",
        loc="left", fontweight="bold", fontsize=9, pad=6
    )
    ax.set_ylim(0, max(counts.values) * 1.25)

    ax.text(
        0.98, 0.95,
        (
            f"A-C (direct malodor/OR evidence): "
            f"{n_direct_or_malodor}/{n_total}\n"
            f"A-D (any related evidence): "
            f"{n_supported}/{n_total}"
        ),
        transform=ax.transAxes,
        ha="right", va="top", fontsize=6.5,
        bbox=dict(
            boxstyle="round,pad=0.28",
            facecolor="white",
            edgecolor="0.7",
            linewidth=0.8,
        )
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(pad=0.7)
    save(fig, outdir, "Figure5B_evidence_grade_distribution", dpi)


def choose_representatives(df):
    # Prefer one representative each from A, B, C:
    # A: best-ranked direct OR antagonist
    # B: sensory malodor suppression
    # C: best-ranked malodor-related patent hit
    reps = []
    for grade in ["A", "B", "C"]:
        z = df[df["grade"] == grade].sort_values("novel_rank")
        if len(z):
            reps.append(z.iloc[0])
    return reps


def panel_c(df, outdir, dpi):
    reps = choose_representatives(df)

    fig, ax = plt.subplots(figsize=PANEL_SIZE)
    ax.axis("off")
    ax.set_title(
        "C  Representative literature/patent-supported candidates",
        loc="left", fontweight="bold", fontsize=9, pad=6
    )

    # Header
    headers = ["Candidate", "Prediction", "Grade", "External evidence"]
    xcols = [0.01, 0.30, 0.48, 0.57]
    for x, h in zip(xcols, headers):
        ax.text(x, 0.90, h, transform=ax.transAxes, fontweight="bold", fontsize=10)

    y_positions = [0.70, 0.44, 0.18]

    for y, row in zip(y_positions, reps):
        name = shorten(row["pubchem_title"], 34)
        rank = int(row["novel_rank"])
        meanp = float(row["mean_predicted_pIC50"])
        minp = float(row["min_predicted_pIC50"])
        grade = str(row["grade"])

        ev = str(row["evidence_summary"])
        ev = textwrap.fill(ev, width=38)

        ax.text(
            xcols[0], y,
            f"{name}\nRank {rank}",
            transform=ax.transAxes,
            va="center", fontsize=6.5
        )
        ax.text(
            xcols[1], y,
            f"Mean pIC50 = {meanp:.3f}\nMin pIC50 = {minp:.3f}",
            transform=ax.transAxes,
            va="center", fontsize=6.5
        )
        ax.text(
            xcols[2], y,
            grade,
            transform=ax.transAxes,
            va="center", ha="center",
            fontsize=8, fontweight="bold",
            bbox=dict(
                boxstyle="circle,pad=0.45",
                facecolor="white",
                edgecolor="0.5",
                linewidth=1.0,
            )
        )
        ax.text(
            xcols[3], y,
            ev,
            transform=ax.transAxes,
            va="center", fontsize=5.8
        )

        # horizontal divider
        if y > y_positions[-1]:
            ax.plot(
                [0.02, 0.98], [y - 0.13, y - 0.13],
                transform=ax.transAxes,
                linewidth=0.7, alpha=0.35
            )

    ax.text(
        0.02, 0.02,
        "Representative hits selected as the highest-ranked examples of grades A, B, and C.",
        transform=ax.transAxes,
        fontsize=6.5,
        style="italic"
    )

    fig.tight_layout(pad=0.7)
    save(fig, outdir, "Figure5C_representative_supported_hits", dpi)


def main():
    args = parse_args()
    df = pd.read_csv(args.input)

    required = [
        "novel_rank", "pubchem_title", "mean_predicted_pIC50",
        "min_predicted_pIC50", "grade", "evidence_summary"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.copy()
    df["grade"] = df["grade"].astype(str).str.strip().str.upper()
    df = df[df["grade"].isin(GRADE_ORDER)].sort_values("novel_rank")

    print("=== Figure 5A-C ===")
    print("Rows:", len(df))
    print("Grade counts:")
    print(df["grade"].value_counts().reindex(GRADE_ORDER, fill_value=0).to_string())

    panel_a(df, args.outdir, args.dpi)
    panel_b(df, args.outdir, args.dpi)
    panel_c(df, args.outdir, args.dpi)

    print("\nRepresentative candidates:")
    for r in choose_representatives(df):
        print(
            f"  Grade {r['grade']}: rank {int(r['novel_rank'])}, "
            f"{r['pubchem_title']}"
        )

    print("\nOutputs written to:", Path(args.outdir).resolve())


if __name__ == "__main__":
    main()
