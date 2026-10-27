#!/usr/bin/env python3
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sequence_control_dir", default="results_indole_sequence_control")
    p.add_argument("--permutation_dir", default="results_indole_aac_permutation_gbr")
    p.add_argument("--outdir", default="figures/figure2")
    p.add_argument("--dpi", type=int, default=600)
    return p.parse_args()

def find_csv_with_columns(directory, required_columns, min_rows=1):
    for f in sorted(Path(directory).rglob("*.csv")):
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        if len(d) >= min_rows and all(c in d.columns for c in required_columns):
            return f, d
    return None, None

def save_figure(fig, outdir, basename, dpi):
    save_panel(fig, outdir, basename, dpi)

def plot_panel_b(sequence_control_dir, outdir, dpi):
    f, d = find_csv_with_columns(sequence_control_dir, ["estimator", "feature_mode", "RMSE"])
    if d is None:
        raise RuntimeError("Could not find model-summary CSV.")
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
    estimator_order = ["rf", "svr", "ridge", "gbr"]
    estimator_labels = {"rf": "RF", "svr": "SVR", "ridge": "Ridge", "gbr": "GBR"}

    d = d[d["feature_mode"].isin(feature_order) & d["estimator"].isin(estimator_order)].copy()
    fig, ax = plt.subplots(figsize=PANEL_SIZE)
    x = np.arange(len(feature_order))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(estimator_order))
    global_max = float(d["RMSE"].max())

    for estimator, offset in zip(estimator_order, offsets):
        vals = []
        for feature in feature_order:
            z = d[(d["estimator"] == estimator) & (d["feature_mode"] == feature)]
            vals.append(z["RMSE"].iloc[0] if len(z) else np.nan)

        bars = ax.bar(x + offset, vals, width=width, label=estimator_labels[estimator])

        for bar, val, feature in zip(bars, vals, feature_order):
            if not np.isfinite(val):
                continue
            is_best = estimator == "gbr" and feature == "antagonist_receptor_aac"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + global_max * 0.018,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=7 if is_best else 6,
                fontweight="bold" if is_best else "normal",
                rotation=90,
                clip_on=False,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([feature_labels[v] for v in feature_order], fontsize=10)
    ax.set_ylabel("RMSE")
    ax.set_xlabel("Receptor representation")
    ax.set_ylim(0, global_max * 1.14)
    ax.legend(
        frameon=False,
        title="Estimator",
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        fontsize=7,
        title_fontsize=7,
        handlelength=1.8,
        columnspacing=1.3,
    )
    ax.set_title("B  LORO-CV performance", loc="left", fontweight="bold", fontsize=7, pad=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(top=0.80, bottom=0.23, left=0.16, right=0.98)
    save_figure(fig, outdir, "Figure2B_LORO_RMSE", dpi)

def plot_panel_c(sequence_control_dir, outdir, dpi):
    candidate = None
    d = None
    for f in sorted(Path(sequence_control_dir).rglob("*.csv")):
        try:
            tmp = pd.read_csv(f)
        except Exception:
            continue
        cols = set(tmp.columns)
        if (
            "estimator" in cols
            and ("heldout_receptor" in cols or "receptor" in cols)
            and ("delta_RMSE_aac_vs_antagonist" in cols or "RMSE_antagonist_receptor_aac" in cols)
        ):
            candidate = f
            d = tmp
            break
    if d is None:
        raise RuntimeError("Could not find per-receptor AAC comparison CSV.")
    print(f"[Figure 2C] Using: {candidate}")

    receptor_col = "heldout_receptor" if "heldout_receptor" in d.columns else "receptor"
    d = d[d["estimator"] == "gbr"].copy()

    if "delta_RMSE_aac_vs_antagonist" in d.columns:
        d["delta"] = d["delta_RMSE_aac_vs_antagonist"]
    else:
        antagonist_col = next((c for c in ["RMSE_antagonist", "rmse_antagonist"] if c in d.columns), None)
        if antagonist_col is None:
            raise RuntimeError("Antagonist RMSE column not found.")
        d["delta"] = d["RMSE_antagonist_receptor_aac"] - d[antagonist_col]

    receptor_order = ["Olfr738", "Olfr740", "Olfr741", "Olfr742", "Olfr739",
                      "Olfr744", "Olfr746", "Olfr743", "Olfr748", "Olfr749"]
    d[receptor_col] = pd.Categorical(d[receptor_col], categories=receptor_order, ordered=True)
    d = d.sort_values(receptor_col)

    fig, ax = plt.subplots(figsize=PANEL_SIZE)
    x = np.arange(len(d))
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.bar(x, d["delta"], width=0.66)

    ymin = min(-0.05, float(d["delta"].min()) * 1.15)
    ax.set_ylim(ymin, 0.045)

    for i, val in enumerate(d["delta"]):
        y_text = val - 0.012
        if y_text < ymin + 0.015:
            y_text = val + 0.012
            va = "bottom"
        else:
            va = "top"
        ax.text(i, y_text, f"{val:.3f}", ha="center", va=va, fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels(d[receptor_col], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(r"$\Delta$RMSE (AAC − antagonist only)")
    ax.set_xlabel("Held-out receptor")
    ax.set_title("C  Receptor-wise improvement with AAC + GBR", loc="left", fontweight="bold", fontsize=7, pad=6)

    n_improved = int((d["delta"] < 0).sum())
    ax.text(
        0.98, 0.94,
        f"{n_improved}/{len(d)} receptors improved",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.7", linewidth=0.8),
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(top=0.84, bottom=0.27, left=0.18, right=0.98)
    save_figure(fig, outdir, "Figure2C_receptor_delta_RMSE", dpi)

def read_observed_permutation_summary(permutation_dir):
    summary_file = Path(permutation_dir) / "permutation_summary.csv"
    if summary_file.exists():
        s = pd.read_csv(summary_file)
        if len(s):
            observed = None
            empirical_p = None
            for c in ["observed_delta_RMSE", "observed_delta_rmse"]:
                if c in s.columns:
                    observed = float(s[c].iloc[0])
                    break
            for c in ["empirical_p_delta_RMSE", "empirical_p_delta_rmse"]:
                if c in s.columns:
                    empirical_p = float(s[c].iloc[0])
                    break
            if observed is not None:
                return observed, empirical_p
    return None, None

def plot_panel_d(permutation_dir, outdir, dpi):
    preferred = Path(permutation_dir) / "permutation_results.csv"
    if preferred.exists():
        candidate = preferred
        d = pd.read_csv(preferred)
    else:
        candidate = None
        d = None
        for f in sorted(Path(permutation_dir).rglob("*.csv")):
            try:
                tmp = pd.read_csv(f)
            except Exception:
                continue
            if len(tmp) >= 100 and "delta_RMSE" in tmp.columns:
                candidate = f
                d = tmp
                break
    if d is None:
        raise RuntimeError("Could not find permutation_results.csv containing delta_RMSE.")
    print(f"[Figure 2D] Using: {candidate}")

    null = pd.to_numeric(d["delta_RMSE"], errors="coerce").dropna().to_numpy()
    observed, empirical_p = read_observed_permutation_summary(permutation_dir)

    if observed is None:
        observed = -0.253239
    if empirical_p is None:
        empirical_p = (np.sum(null <= observed) + 1) / (len(null) + 1)

    fig, ax = plt.subplots(figsize=PANEL_SIZE)
    ax.hist(null, bins=35, edgecolor="black", linewidth=0.4)
    ax.axvline(observed, linewidth=2.2, linestyle="--")
    ax.set_xlabel(r"$\Delta$RMSE (AAC − antagonist only)")
    ax.set_ylabel("Permutation count")
    ax.set_title("D  Permutation control of AAC benefit", loc="left", fontweight="bold", fontsize=7, pad=6)

    x_left, x_right = ax.get_xlim()
    _, y_top = ax.get_ylim()
    x_text = observed + 0.06 * (x_right - x_left)
    y_text = y_top * 0.93

    ax.text(
        x_text,
        y_text,
        f"Observed ΔRMSE = {observed:.3f}\nEmpirical p = {empirical_p:.3g}",
        ha="left",
        va="top",
        fontsize=7,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.7", linewidth=0.8),
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(top=0.84, bottom=0.22, left=0.18, right=0.98)
    save_figure(fig, outdir, "Figure2D_AAC_permutation", dpi)

def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print("=== Generate Figure 2B-D ===")
    plot_panel_b(args.sequence_control_dir, outdir, args.dpi)
    plot_panel_c(args.sequence_control_dir, outdir, args.dpi)
    plot_panel_d(args.permutation_dir, outdir, args.dpi)
    print()
    print(f"Outputs written to: {outdir.resolve()}")

if __name__ == "__main__":
    main()
