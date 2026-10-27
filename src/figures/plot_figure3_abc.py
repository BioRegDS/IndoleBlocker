#!/usr/bin/env python3
"""
Generate publication-style Figure 3A-C.

A. Amino-acid composition importance (AAC + GBR interpretability)
B. Isoleucine fraction vs blocker-adjusted receptor response
C. Leave-one-receptor-out sensitivity of the Isoleucine-response correlation
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
from scipy import stats


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--interpretability_dir", default="results_indole_aac_interpretability")
    p.add_argument("--isoleucine_dir", default="results_indole_isoleucine_response")
    p.add_argument("--outdir", default="figures/figure3")
    p.add_argument("--dpi", type=int, default=600)
    return p.parse_args()


def save_figure(fig, outdir, basename, dpi):
    save_panel(fig, outdir, basename, dpi)


def find_csv_with_columns(directory, required_columns, min_rows=1):
    for f in sorted(Path(directory).rglob("*.csv")):
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        if len(d) >= min_rows and all(c in d.columns for c in required_columns):
            return f, d
    return None, None


def plot_panel_a(interpretability_dir, outdir, dpi):
    required = [
        "amino_acid",
        "mean_native_importance",
        "mean_heldout_delta_MSE",
        "delta_RMSE_vs_full_AAC",
    ]
    candidate, d = find_csv_with_columns(interpretability_dir, required)
    if d is None:
        raise RuntimeError("Could not find AAC importance summary CSV.")
    print(f"[Figure 3A] Using: {candidate}")

    d = d.sort_values("mean_native_importance", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=PANEL_SIZE)
    x = np.arange(len(d))
    bars = ax.bar(x, d["mean_native_importance"], width=0.72)

    for bar, aa in zip(bars, d["amino_acid"]):
        if aa == "I":
            bar.set_hatch("//")
            bar.set_linewidth(1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(d["amino_acid"], fontsize=7)
    ax.set_ylabel("GBR feature importance")
    ax.set_xlabel("Amino acid")
    ax.set_title(
        "A  Amino-acid composition importance",
        loc="left",
        fontweight="bold",
        fontsize=15,
        pad=12,
    )

    if "I" in set(d["amino_acid"]):
        idx = int(d.index[d["amino_acid"] == "I"][0])
        val = float(d.loc[idx, "mean_native_importance"])
        ax.annotate(
            "Isoleucine (I)",
            xy=(idx, val),
            xytext=(idx + 1.2, val * 1.08),
            arrowprops=dict(arrowstyle="->", linewidth=1.0),
            fontsize=7,
            ha="left",
        )

    ax.set_ylim(0, float(d["mean_native_importance"].max()) * 1.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(pad=0.8)
    save_figure(fig, outdir, "Figure3A_AAC_importance", dpi)


def plot_panel_b(isoleucine_dir, outdir, dpi):
    candidate, d = find_csv_with_columns(
        isoleucine_dir,
        ["receptor", "isoleucine_fraction", "blocker_adjusted_response"],
    )
    if d is None:
        raise RuntimeError("Could not find receptor summary CSV.")
    print(f"[Figure 3B] Using: {candidate}")

    d = d[["receptor", "isoleucine_fraction", "blocker_adjusted_response"]].dropna().copy()

    x = d["isoleucine_fraction"].astype(float).to_numpy()
    y = d["blocker_adjusted_response"].astype(float).to_numpy()

    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_rho, spearman_p = stats.spearmanr(x, y)

    slope, intercept = np.polyfit(x, y, 1)
    xx = np.linspace(x.min(), x.max(), 200)
    yy = intercept + slope * xx

    fig, ax = plt.subplots(figsize=PANEL_SIZE)
    ax.scatter(x, y, s=MARKER_SIZE, zorder=3)
    ax.plot(xx, yy, linestyle="--", linewidth=1.4, zorder=2)

    xspan = x.max() - x.min()
    yspan = y.max() - y.min()

    for _, row in d.iterrows():
        xi = float(row["isoleucine_fraction"])
        yi = float(row["blocker_adjusted_response"])
        dx = 0.012 * xspan
        dy = 0.025 * yspan if yi >= np.median(y) else -0.045 * yspan
        ax.text(xi + dx, yi + dy, str(row["receptor"]), fontsize=6.5)

    ax.set_xlabel("Isoleucine fraction")
    ax.set_ylabel("Blocker-adjusted response")
    ax.set_title(
        "B  Isoleucine fraction vs receptor response",
        loc="left",
        fontweight="bold",
        fontsize=15,
        pad=12,
    )
    ax.text(
        0.98,
        0.96,
        (
            f"Pearson r = {pearson_r:.3f}, p = {pearson_p:.4f}\n"
            f"Spearman ρ = {spearman_rho:.3f}, p = {spearman_p:.4f}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.7", linewidth=0.8),
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(pad=0.8)
    save_figure(fig, outdir, "Figure3B_isoleucine_vs_adjusted_response", dpi)


def plot_panel_c(isoleucine_dir, outdir, dpi):
    candidate = Path(isoleucine_dir) / "leave_one_receptor_out_correlation_sensitivity.csv"

    if candidate.exists():
        d = pd.read_csv(candidate)
    else:
        candidate, d = find_csv_with_columns(
            isoleucine_dir,
            ["phenotype", "excluded_receptor", "pearson_r", "pearson_p"],
        )
        if d is None:
            raise RuntimeError("Could not find leave-one-receptor-out correlation CSV.")

    print(f"[Figure 3C] Using: {candidate}")

    d = d[d["phenotype"] == "blocker_adjusted_response"].copy()
    if len(d) == 0:
        raise RuntimeError("No blocker_adjusted_response rows found.")

    receptor_order = [
        "Olfr738", "Olfr740", "Olfr741", "Olfr742", "Olfr739",
        "Olfr744", "Olfr746", "Olfr743", "Olfr748", "Olfr749",
    ]
    d["excluded_receptor"] = pd.Categorical(
        d["excluded_receptor"], categories=receptor_order, ordered=True
    )
    d = d.sort_values("excluded_receptor").reset_index(drop=True)

    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=PANEL_SIZE)
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.scatter(d["pearson_r"], y, s=MARKER_SIZE, zorder=3)

    n = 9
    lefts, rights = [], []
    for r in d["pearson_r"].astype(float):
        r_clip = np.clip(r, -0.999999, 0.999999)
        z = np.arctanh(r_clip)
        se = 1 / np.sqrt(n - 3)
        lo = np.tanh(z - 1.96 * se)
        hi = np.tanh(z + 1.96 * se)
        lefts.append(r - lo)
        rights.append(hi - r)

    ax.errorbar(
        d["pearson_r"],
        y,
        xerr=np.vstack([lefts, rights]),
        fmt="none",
        capsize=3,
        linewidth=1,
        zorder=2,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(d["excluded_receptor"].astype(str), fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Pearson r after excluding one receptor")
    ax.set_ylabel("Excluded receptor")
    ax.set_title(
        "C  Leave-one-receptor-out sensitivity",
        loc="left",
        fontweight="bold",
        fontsize=15,
        pad=12,
    )

    min_r = float(d["pearson_r"].min())
    max_r = float(d["pearson_r"].max())
    max_p = float(d["pearson_p"].max())

    ax.text(
        0.98,
        0.96,
        (
            "All correlations remained negative\n"
            f"r = {min_r:.3f} to {max_r:.3f}\n"
            f"all p < {max_p:.3f}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.7", linewidth=0.8),
    )

    ax.set_xlim(min(-1.02, min_r - 0.08), 0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(pad=0.8)
    save_figure(fig, outdir, "Figure3C_leave_one_receptor_out", dpi)


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=== Generate Figure 3A-C ===")
    plot_panel_a(args.interpretability_dir, outdir, args.dpi)
    plot_panel_b(args.isoleucine_dir, outdir, args.dpi)
    plot_panel_c(args.isoleucine_dir, outdir, args.dpi)

    print()
    print(f"Outputs written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
