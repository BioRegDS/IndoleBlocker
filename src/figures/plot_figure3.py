#!/usr/bin/env python3
"""
Generate composite Figure 3.

Layout
------
A: upper-left, amino-acid composition importance.
B: upper-right, isoleucine fraction vs blocker-adjusted receptor response.
C: lower-left, leave-one-receptor-out sensitivity analysis.
D: lower-right, intentionally left blank for optional manuscript schematic
   or future panel.

The full figure is sized to match the compressed two-column Figure 2 layout.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--interpretability_dir",
        default="results/interpretability/aac",
    )
    p.add_argument(
        "--isoleucine_dir",
        default="results/interpretability/isoleucine",
    )
    p.add_argument(
        "--outdir",
        default="figures/figure3",
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


def find_csv_with_columns(directory, required_columns, min_rows=1):
    for f in sorted(Path(directory).rglob("*.csv")):
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        if len(d) >= min_rows and all(c in d.columns for c in required_columns):
            return f, d
    return None, None


# ---------------------------------------------------------------------
# Panel A
# ---------------------------------------------------------------------
def draw_panel_a(ax, interpretability_dir):
    required = [
        "amino_acid",
        "mean_native_importance",
        "mean_heldout_delta_MSE",
        "delta_RMSE_vs_full_AAC",
    ]
    candidate, d = find_csv_with_columns(
        interpretability_dir,
        required,
    )
    if d is None:
        raise RuntimeError(
            "Could not find AAC importance summary CSV."
        )

    print(f"[Figure 3A] Using: {candidate}")

    d = (
        d.sort_values(
            "mean_native_importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    x = np.arange(len(d))
    bars = ax.bar(
        x,
        d["mean_native_importance"],
        width=0.72,
    )

    for bar, aa in zip(
        bars,
        d["amino_acid"],
    ):
        if aa == "I":
            bar.set_hatch("//")
            bar.set_linewidth(1.2)

    ax.set_xticks(x)
    ax.set_xticklabels(
        d["amino_acid"],
        fontsize=5.8,
    )

    ax.set_ylabel(
        "GBR feature importance"
    )
    ax.set_xlabel(
        "Amino acid"
    )

    panel_title(
        ax,
        "A",
        "Amino-acid composition importance",
    )

    if "I" in set(d["amino_acid"]):
        idx = int(
            d.index[
                d["amino_acid"] == "I"
            ][0]
        )
        val = float(
            d.loc[
                idx,
                "mean_native_importance",
            ]
        )

        ax.annotate(
            "Isoleucine",
            xy=(idx, val),
            xytext=(idx + 2.0, val * 1.08),
            arrowprops=dict(
                arrowstyle="->",
                linewidth=0.8,
            ),
            fontsize=6.2,
            ha="left",
        )

    ax.set_ylim(
        0,
        float(
            d["mean_native_importance"].max()
        ) * 1.22,
    )

    style_axes(ax)


# ---------------------------------------------------------------------
# Panel B
# ---------------------------------------------------------------------
def draw_panel_b(ax, isoleucine_dir):
    candidate, d = find_csv_with_columns(
        isoleucine_dir,
        [
            "receptor",
            "isoleucine_fraction",
            "blocker_adjusted_response",
        ],
    )

    if d is None:
        raise RuntimeError(
            "Could not find receptor summary CSV."
        )

    print(f"[Figure 3B] Using: {candidate}")

    d = (
        d[
            [
                "receptor",
                "isoleucine_fraction",
                "blocker_adjusted_response",
            ]
        ]
        .dropna()
        .copy()
    )

    x = (
        d["isoleucine_fraction"]
        .astype(float)
        .to_numpy()
    )
    y = (
        d["blocker_adjusted_response"]
        .astype(float)
        .to_numpy()
    )

    pearson_r, pearson_p = stats.pearsonr(
        x,
        y,
    )
    spearman_rho, spearman_p = stats.spearmanr(
        x,
        y,
    )

    slope, intercept = np.polyfit(
        x,
        y,
        1,
    )

    xx = np.linspace(
        x.min(),
        x.max(),
        200,
    )
    yy = intercept + slope * xx

    ax.scatter(
        x,
        y,
        s=24,
        zorder=3,
    )

    ax.plot(
        xx,
        yy,
        linestyle="--",
        linewidth=1.0,
        zorder=2,
    )

    # Receptor-specific label offsets to avoid overlap.
    xspan = x.max() - x.min()
    yspan = y.max() - y.min()

    label_offsets = {
        "Olfr738": (0.008,  0.060),
        "Olfr739": (0.010,  0.045),
        "Olfr740": (0.010, -0.070),
        "Olfr741": (0.012, -0.055),
        "Olfr742": (0.010, -0.050),
        "Olfr743": (0.014,  0.095),
        "Olfr744": (0.014, -0.065),
        "Olfr746": (0.010, -0.055),
        "Olfr748": (0.012, -0.060),
        "Olfr749": (0.012, -0.050),
    }

    for _, row in d.iterrows():
        xi = float(row["isoleucine_fraction"])
        yi = float(row["blocker_adjusted_response"])
        receptor = str(row["receptor"])

        dx_frac, dy_frac = label_offsets.get(
            receptor,
            (0.010, 0.040),
        )

        ax.text(
            xi + dx_frac * xspan,
            yi + dy_frac * yspan,
            receptor,
            fontsize=5.7,
            ha="left",
            va="center",
        )

    ax.set_xlabel(
        "Isoleucine fraction"
    )
    ax.set_ylabel(
        "Blocker-adjusted response"
    )

    panel_title(
        ax,
        "B",
        "Isoleucine fraction vs response",
    )

    ax.text(
        0.97,
        0.95,
        (
            f"Pearson r = {pearson_r:.3f}, p = {pearson_p:.4f}\n"
            f"Spearman ρ = {spearman_rho:.3f}, p = {spearman_p:.4f}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.0,
        bbox=dict(
            boxstyle="round,pad=0.18",
            facecolor="white",
            edgecolor="none",
            alpha=0.85,
        ),
    )

    style_axes(ax)


# ---------------------------------------------------------------------
# Panel C
# ---------------------------------------------------------------------
def draw_panel_c(ax, isoleucine_dir):
    candidate = (
        Path(isoleucine_dir)
        / "leave_one_receptor_out_correlation_sensitivity.csv"
    )

    if candidate.exists():
        d = pd.read_csv(candidate)
    else:
        candidate, d = find_csv_with_columns(
            isoleucine_dir,
            [
                "phenotype",
                "excluded_receptor",
                "pearson_r",
                "pearson_p",
            ],
        )

        if d is None:
            raise RuntimeError(
                "Could not find leave-one-receptor-out correlation CSV."
            )

    print(f"[Figure 3C] Using: {candidate}")

    d = d[
        d["phenotype"]
        == "blocker_adjusted_response"
    ].copy()

    if len(d) == 0:
        raise RuntimeError(
            "No blocker_adjusted_response rows found."
        )

    receptor_order = [
        "Olfr738",
        "Olfr739",
        "Olfr740",
        "Olfr741",
        "Olfr742",
        "Olfr743",
        "Olfr744",
        "Olfr746",
        "Olfr748",
        "Olfr749",
    ]

    d["excluded_receptor"] = pd.Categorical(
        d["excluded_receptor"],
        categories=receptor_order,
        ordered=True,
    )

    d = (
        d.sort_values(
            "excluded_receptor"
        )
        .reset_index(drop=True)
    )

    y = np.arange(len(d))

    ax.axvline(
        0,
        linestyle="--",
        linewidth=0.8,
    )

    ax.scatter(
        d["pearson_r"],
        y,
        s=22,
        zorder=3,
    )

    # 95% CI using Fisher z-transform, n=9 after exclusion.
    n = 9
    lefts = []
    rights = []

    for r in d["pearson_r"].astype(float):
        r_clip = np.clip(
            r,
            -0.999999,
            0.999999,
        )
        z = np.arctanh(r_clip)
        se = 1 / np.sqrt(n - 3)

        lo = np.tanh(
            z - 1.96 * se
        )
        hi = np.tanh(
            z + 1.96 * se
        )

        lefts.append(
            r - lo
        )
        rights.append(
            hi - r
        )

    ax.errorbar(
        d["pearson_r"],
        y,
        xerr=np.vstack(
            [lefts, rights]
        ),
        fmt="none",
        capsize=2,
        linewidth=0.8,
        zorder=2,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(
        d["excluded_receptor"].astype(str),
        fontsize=6.0,
    )

    ax.invert_yaxis()

    ax.set_xlabel(
        "Pearson r after excluding one receptor"
    )
    ax.set_ylabel(
        "Excluded receptor"
    )

    panel_title(
        ax,
        "C",
        "Leave-one-receptor-out sensitivity",
    )

    min_r = float(
        d["pearson_r"].min()
    )
    max_r = float(
        d["pearson_r"].max()
    )
    max_p = float(
        d["pearson_p"].max()
    )

    ax.text(
        0.86,
        0.08,
        (
            "All correlations negative\n"
            f"r = {min_r:.3f} to {max_r:.3f}\n"
            f"all p < {max_p:.3f}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.0,
        bbox=dict(
            boxstyle="round,pad=0.18",
            facecolor="white",
            edgecolor="none",
            alpha=0.85,
        ),
    )

    ax.set_xlim(
        min(
            -1.02,
            min_r - 0.08,
        ),
        0.05,
    )

    style_axes(ax)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    args = parse_args()

    print(
        "=== Generate composite Figure 3 ==="
    )

    fig = plt.figure(
        figsize=FIGSIZE,
    )

    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        height_ratios=[1.0, 0.95],
        width_ratios=[1.0, 1.0],
        hspace=0.42,
        wspace=0.36,
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
        args.interpretability_dir,
    )

    draw_panel_b(
        ax_b,
        args.isoleucine_dir,
    )

    draw_panel_c(
        ax_c,
        args.isoleucine_dir,
    )

    fig.subplots_adjust(
        left=0.10,
        right=0.97,
        bottom=0.11,
        top=0.96,
    )

    save(
        fig,
        args.outdir,
        "Figure3_composite",
        args.dpi,
    )

    print(
        "Outputs written to:",
        Path(args.outdir).resolve(),
    )


if __name__ == "__main__":
    main()
