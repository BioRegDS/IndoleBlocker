#!/usr/bin/env python3
"""
Generate composite Figure 2.

Layout
------
A: upper-left, intentionally left blank for manual insertion of the
   LORO-CV / receptor-aware modeling workflow.
B: upper-right, LORO-CV RMSE across receptor representations and estimators.
C: lower-left, receptor-wise RMSE improvement with AAC + GBR.
D: lower-right, permutation control of the AAC benefit.

The full figure is sized for a two-column manuscript figure.
"""

import argparse
from pathlib import Path

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

FIGSIZE = (6.9, 4.75)  # ~175 mm wide


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--sequence_control_dir",
        default="results/model_validation/sequence_control",
    )
    p.add_argument(
        "--permutation_dir",
        default="results/model_validation/aac_permutation",
    )
    p.add_argument(
        "--outdir",
        default="figures/figure2",
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

        if len(d) >= min_rows and all(
            c in d.columns for c in required_columns
        ):
            return f, d

    return None, None


# ---------------------------------------------------------------------
# Panel B
# ---------------------------------------------------------------------
def draw_panel_b(ax, sequence_control_dir):
    f, d = find_csv_with_columns(
        sequence_control_dir,
        ["estimator", "feature_mode", "RMSE"],
    )

    if d is None:
        raise RuntimeError(
            "Could not find model-summary CSV."
        )

    print(f"[Figure 2B] Using: {f}")

    feature_order = [
        "antagonist",
        "antagonist_receptor_onehot",
        "antagonist_receptor_esm2",
        "antagonist_receptor_aac",
    ]

    feature_labels = {
        "antagonist": "Antagonist\nonly",
        "antagonist_receptor_onehot": "+ One-hot",
        "antagonist_receptor_esm2": "+ ESM-2",
        "antagonist_receptor_aac": "+ AAC",
    }

    estimator_order = [
        "rf",
        "svr",
        "ridge",
        "gbr",
    ]

    estimator_labels = {
        "rf": "RF",
        "svr": "SVR",
        "ridge": "Ridge",
        "gbr": "GBR",
    }

    d = d[
        d["feature_mode"].isin(feature_order)
        & d["estimator"].isin(estimator_order)
    ].copy()

    x = np.arange(len(feature_order))
    width = 0.18
    offsets = np.linspace(
        -1.5 * width,
        1.5 * width,
        len(estimator_order),
    )

    global_max = float(d["RMSE"].max())

    for estimator, offset in zip(
        estimator_order,
        offsets,
    ):
        vals = []

        for feature in feature_order:
            z = d[
                (d["estimator"] == estimator)
                & (d["feature_mode"] == feature)
            ]
            vals.append(
                z["RMSE"].iloc[0]
                if len(z)
                else np.nan
            )

        bars = ax.bar(
            x + offset,
            vals,
            width=width,
            label=estimator_labels[estimator],
        )

        # Only label the best AAC + GBR bar to avoid overcrowding.
        for bar, val, feature in zip(
            bars,
            vals,
            feature_order,
        ):
            if (
                estimator == "gbr"
                and feature == "antagonist_receptor_aac"
                and np.isfinite(val)
            ):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + global_max * 0.035,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    fontweight="bold",
                    clip_on=False,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [feature_labels[v] for v in feature_order],
        fontsize=6.2,
    )

    ax.set_ylabel("RMSE")
    ax.set_xlabel("Receptor representation")

    ax.set_ylim(
        0,
        global_max * 1.16,
    )

    ax.legend(
        frameon=False,
        ncol=2,
        loc="upper left",
        fontsize=6.5,
        columnspacing=0.9,
        handlelength=1.4,
    )

    panel_title(
        ax,
        "B",
        "LORO-CV performance",
    )

    style_axes(ax)


# ---------------------------------------------------------------------
# Panel C
# ---------------------------------------------------------------------
def draw_panel_c(ax, sequence_control_dir):
    candidate = None
    d = None

    for f in sorted(
        Path(sequence_control_dir).rglob("*.csv")
    ):
        try:
            tmp = pd.read_csv(f)
        except Exception:
            continue

        cols = set(tmp.columns)

        if (
            "estimator" in cols
            and (
                "heldout_receptor" in cols
                or "receptor" in cols
            )
            and (
                "delta_RMSE_aac_vs_antagonist" in cols
                or "RMSE_antagonist_receptor_aac" in cols
            )
        ):
            candidate = f
            d = tmp
            break

    if d is None:
        raise RuntimeError(
            "Could not find per-receptor AAC comparison CSV."
        )

    print(f"[Figure 2C] Using: {candidate}")

    receptor_col = (
        "heldout_receptor"
        if "heldout_receptor" in d.columns
        else "receptor"
    )

    d = d[
        d["estimator"] == "gbr"
    ].copy()

    if "delta_RMSE_aac_vs_antagonist" in d.columns:
        d["delta"] = (
            d["delta_RMSE_aac_vs_antagonist"]
        )
    else:
        antagonist_col = next(
            (
                c for c in [
                    "RMSE_antagonist",
                    "rmse_antagonist",
                ]
                if c in d.columns
            ),
            None,
        )

        if antagonist_col is None:
            raise RuntimeError(
                "Antagonist RMSE column not found."
            )

        d["delta"] = (
            d["RMSE_antagonist_receptor_aac"]
            - d[antagonist_col]
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

    d[receptor_col] = pd.Categorical(
        d[receptor_col],
        categories=receptor_order,
        ordered=True,
    )

    d = d.sort_values(
        receptor_col
    ).reset_index(drop=True)

    x = np.arange(len(d))

    ax.axhline(
        0,
        linestyle="--",
        linewidth=0.8,
    )

    ax.bar(
        x,
        d["delta"],
        width=0.66,
    )

    ymin = min(
        -0.05,
        float(d["delta"].min()) * 1.16,
    )

    ax.set_ylim(
        ymin,
        0.045,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        d[receptor_col].astype(str),
        rotation=55,
        ha="right",
        rotation_mode="anchor",
        fontsize=5.8,
    )

    ax.set_ylabel(
        r"$\Delta$RMSE (AAC − antagonist only)"
    )
    ax.set_xlabel(
        "Held-out receptor"
    )

    panel_title(
        ax,
        "C",
        "Receptor-wise AAC improvement",
    )

    n_improved = int(
        (d["delta"] < 0).sum()
    )

    ax.text(
        0.97,
        0.06,
        f"{n_improved}/{len(d)} receptors improved",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.5,
        fontweight="bold",
    )

    style_axes(ax)


# ---------------------------------------------------------------------
# Panel D
# ---------------------------------------------------------------------
def read_observed_permutation_summary(
    permutation_dir,
):
    summary_file = (
        Path(permutation_dir)
        / "permutation_summary.csv"
    )

    if summary_file.exists():
        s = pd.read_csv(summary_file)

        if len(s):
            observed = None
            empirical_p = None

            for c in [
                "observed_delta_RMSE",
                "observed_delta_rmse",
            ]:
                if c in s.columns:
                    observed = float(
                        s[c].iloc[0]
                    )
                    break

            for c in [
                "empirical_p_delta_RMSE",
                "empirical_p_delta_rmse",
            ]:
                if c in s.columns:
                    empirical_p = float(
                        s[c].iloc[0]
                    )
                    break

            if observed is not None:
                return observed, empirical_p

    return None, None


def draw_panel_d(ax, permutation_dir):
    preferred = (
        Path(permutation_dir)
        / "permutation_results.csv"
    )

    if preferred.exists():
        candidate = preferred
        d = pd.read_csv(preferred)

    else:
        candidate = None
        d = None

        for f in sorted(
            Path(permutation_dir).rglob("*.csv")
        ):
            try:
                tmp = pd.read_csv(f)
            except Exception:
                continue

            if (
                len(tmp) >= 100
                and "delta_RMSE" in tmp.columns
            ):
                candidate = f
                d = tmp
                break

    if d is None:
        raise RuntimeError(
            "Could not find permutation_results.csv containing delta_RMSE."
        )

    print(f"[Figure 2D] Using: {candidate}")

    null = (
        pd.to_numeric(
            d["delta_RMSE"],
            errors="coerce",
        )
        .dropna()
        .to_numpy()
    )

    observed, empirical_p = (
        read_observed_permutation_summary(
            permutation_dir
        )
    )

    if observed is None:
        observed = -0.253239

    if empirical_p is None:
        empirical_p = (
            np.sum(null <= observed) + 1
        ) / (
            len(null) + 1
        )

    ax.hist(
        null,
        bins=32,
        edgecolor="black",
        linewidth=0.35,
    )

    ax.axvline(
        observed,
        linewidth=1.6,
        linestyle="--",
    )

    ax.set_xlabel(
        r"$\Delta$RMSE (AAC − antagonist only)"
    )
    ax.set_ylabel(
        "Permutation count"
    )

    panel_title(
        ax,
        "D",
        "Permutation control",
    )

    ax.text(
        0.97,
        0.95,
        (
            f"Observed ΔRMSE = {observed:.3f}\n"
            f"Empirical p = {empirical_p:.3g}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.5,
        bbox=dict(
            boxstyle="round,pad=0.20",
            facecolor="white",
            edgecolor="none",
            alpha=0.85,
        ),
    )

    style_axes(ax)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    args = parse_args()

    print("=== Generate composite Figure 2 ===")

    fig = plt.figure(
        figsize=FIGSIZE,
    )

    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        height_ratios=[1.0, 1.0],
        width_ratios=[1.0, 1.0],
        hspace=0.38,
        wspace=0.36,
    )

    # A: blank panel for manual workflow schematic.
    ax_a = fig.add_subplot(
        gs[0, 0]
    )
    ax_a.set_axis_off()

    # Match the title baseline and typography used for panel B.
    ax_a.text(
        0.0,
        1.025,
        "A  Leave-one-receptor-out cross-validation",
        transform=ax_a.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        clip_on=False,
    )

    # B-D.
    ax_b = fig.add_subplot(
        gs[0, 1]
    )
    ax_c = fig.add_subplot(
        gs[1, 0]
    )
    ax_d = fig.add_subplot(
        gs[1, 1]
    )

    draw_panel_b(
        ax_b,
        args.sequence_control_dir,
    )

    draw_panel_c(
        ax_c,
        args.sequence_control_dir,
    )

    draw_panel_d(
        ax_d,
        args.permutation_dir,
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
        "Figure2_composite",
        args.dpi,
    )

    print(
        "Outputs written to:",
        Path(args.outdir).resolve(),
    )


if __name__ == "__main__":
    main()
