#!/usr/bin/env python3
"""
Isoleucine composition vs receptor response phenotype (Indole-only).

Uses the strict IC50 dataset and the same RDKit blocker standardization used
in the modeling scripts.

For each receptor:
  - Isoleucine fraction (I / standard amino-acid count)
  - mean pIC50
  - median pIC50
  - blocker-adjusted response phenotype

Blocker adjustment:
  For each standardized blocker, subtract the mean pIC50 of that blocker
  across all receptors in which it was measured.
  receptor_adjusted_response = mean of these blocker-centered residuals
  within each receptor.

This removes the first-order effect of which blockers happened to be tested
for each receptor, without requiring a complete receptor x blocker matrix.

Also performs:
  - Pearson and Spearman correlations
  - leave-one-receptor-out sensitivity analysis for the correlations
  - labeled scatter plots
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from scipy.stats import pearsonr, spearmanr


AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
AA_SET = set(AA_ORDER)


def standardize_smiles(s):
    if pd.isna(s):
        return None
    try:
        mol = Chem.MolFromSmiles(str(s))
        if mol is None:
            return None
        mol = rdMolStandardize.FragmentParent(mol)
        mol = rdMolStandardize.Normalizer().normalize(mol)
        return Chem.MolToSmiles(
            mol,
            canonical=True,
            isomericSmiles=True
        )
    except Exception:
        return None


def clean_sequence(x):
    if pd.isna(x):
        return None
    s = "".join(str(x).split()).upper()
    if not s:
        return None

    valid = sum(
        c in AA_SET or c in "XBZUO"
        for c in s
    ) / len(s)

    return s if valid >= 0.95 else None


def isoleucine_fraction(seq):
    standard_count = sum(seq.count(aa) for aa in AA_ORDER)
    if standard_count == 0:
        return np.nan
    return seq.count("I") / standard_count


def safe_corr(x, y, method):
    q = pd.DataFrame({"x": x, "y": y}).dropna()

    if len(q) < 3:
        return np.nan, np.nan, len(q)

    if q["x"].nunique() < 2 or q["y"].nunique() < 2:
        return np.nan, np.nan, len(q)

    if method == "pearson":
        r, p = pearsonr(q["x"], q["y"])
    elif method == "spearman":
        r, p = spearmanr(q["x"], q["y"])
    else:
        raise ValueError(method)

    return float(r), float(p), len(q)


def make_scatter(df, y_col, y_label, title, outpath):
    q = df[[
        "receptor",
        "isoleucine_fraction",
        y_col
    ]].dropna().copy()

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        q["isoleucine_fraction"],
        q[y_col],
        s=70
    )

    # Linear visual guide only.
    if len(q) >= 2 and q["isoleucine_fraction"].nunique() > 1:
        coef = np.polyfit(
            q["isoleucine_fraction"],
            q[y_col],
            1
        )
        xx = np.linspace(
            q["isoleucine_fraction"].min(),
            q["isoleucine_fraction"].max(),
            100
        )
        ax.plot(
            xx,
            coef[0] * xx + coef[1],
            linewidth=1
        )

    for _, row in q.iterrows():
        ax.annotate(
            row["receptor"],
            (
                row["isoleucine_fraction"],
                row[y_col]
            ),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=9
        )

    pr, pp, n = safe_corr(
        q["isoleucine_fraction"],
        q[y_col],
        "pearson"
    )

    sr, sp, _ = safe_corr(
        q["isoleucine_fraction"],
        q[y_col],
        "spearman"
    )

    text = (
        f"Pearson r = {pr:.3f}, p = {pp:.4g}\n"
        f"Spearman rho = {sr:.3f}, p = {sp:.4g}\n"
        f"n = {n} receptors"
    )

    ax.text(
        0.04,
        0.96,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top"
    )

    ax.set_xlabel(
        "Isoleucine fraction in receptor sequence"
    )

    ax.set_ylabel(
        y_label
    )

    ax.set_title(
        title
    )

    fig.tight_layout()

    fig.savefig(
        outpath,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


def leave_one_receptor_out_correlations(summary_df, y_col):
    rows = []

    receptors = summary_df["receptor"].tolist()

    for heldout in receptors:
        q = summary_df[
            summary_df["receptor"] != heldout
        ]

        pr, pp, n = safe_corr(
            q["isoleucine_fraction"],
            q[y_col],
            "pearson"
        )

        sr, sp, _ = safe_corr(
            q["isoleucine_fraction"],
            q[y_col],
            "spearman"
        )

        rows.append({
            "phenotype": y_col,
            "excluded_receptor": heldout,
            "n_receptors": n,
            "pearson_r": pr,
            "pearson_p": pp,
            "spearman_rho": sr,
            "spearman_p": sp
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True
    )

    parser.add_argument(
        "--sheet",
        default="regression_strict_exact_113"
    )

    parser.add_argument(
        "--outdir",
        default="results_indole_isoleucine_response"
    )

    parser.add_argument(
        "--sequence_col",
        default="receptor_sequence"
    )

    args = parser.parse_args()

    outdir = Path(args.outdir)

    outdir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------
    # Load Indole-only IC50 records
    # --------------------------------------------------
    df = pd.read_csv(str(args.input))

    if "endpoint_type" in df.columns:
        df = df[
            df["endpoint_type"]
            .astype(str)
            .str.upper()
            == "IC50"
        ].copy()

    df = df[
        df["malodor"]
        .astype(str)
        .str.strip()
        .str.lower()
        == "indole"
    ].copy()

    if "activity_uM" not in df.columns:
        raise ValueError(
            "Expected activity_uM column in input workbook."
        )

    df["IC50_uM_numeric"] = pd.to_numeric(
        df["activity_uM"],
        errors="coerce"
    )

    df["blocker_SMILES_std"] = (
        df["blocker_SMILES"]
        .apply(standardize_smiles)
    )

    df["receptor_sequence_clean"] = (
        df[args.sequence_col]
        .apply(clean_sequence)
    )

    df = df.dropna(
        subset=[
            "receptor",
            "blocker_SMILES_std",
            "receptor_sequence_clean",
            "IC50_uM_numeric"
        ]
    ).copy()

    df = df[
        df["IC50_uM_numeric"] > 0
    ].reset_index(
        drop=True
    )

    df["receptor"] = (
        df["receptor"]
        .astype(str)
        .str.strip()
    )

    df["pIC50"] = (
        6.0
        - np.log10(
            df["IC50_uM_numeric"]
        )
    )

    # --------------------------------------------------
    # Check one sequence per receptor
    # --------------------------------------------------
    seq_check = (
        df[
            [
                "receptor",
                "receptor_sequence_clean"
            ]
        ]
        .drop_duplicates()
    )

    counts = (
        seq_check
        .groupby("receptor")
        ["receptor_sequence_clean"]
        .nunique()
    )

    bad = counts[
        counts > 1
    ]

    if len(bad):
        raise ValueError(
            "Multiple sequences found for receptor(s): "
            + ", ".join(
                bad.index.astype(str)
            )
        )

    receptor_seq = (
        seq_check
        .drop_duplicates(
            subset=["receptor"]
        )
        .copy()
    )

    receptor_seq["isoleucine_fraction"] = (
        receptor_seq[
            "receptor_sequence_clean"
        ]
        .apply(isoleucine_fraction)
    )

    receptor_seq["sequence_length"] = (
        receptor_seq[
            "receptor_sequence_clean"
        ]
        .str.len()
    )

    # --------------------------------------------------
    # Blocker-adjusted response
    # --------------------------------------------------
    # Average duplicates first if any receptor-blocker
    # pair has more than one exact IC50 record.
    pair_df = (
        df
        .groupby(
            [
                "receptor",
                "blocker_SMILES_std"
            ],
            as_index=False
        )
        .agg(
            pIC50=("pIC50", "mean"),
            blocker=("blocker", "first")
        )
    )

    # Mean response for each blocker across all receptors
    blocker_mean = (
        pair_df
        .groupby(
            "blocker_SMILES_std"
        )["pIC50"]
        .mean()
        .rename("blocker_mean_pIC50")
    )

    pair_df = pair_df.merge(
        blocker_mean,
        on="blocker_SMILES_std",
        how="left"
    )

    pair_df["blocker_centered_pIC50"] = (
        pair_df["pIC50"]
        - pair_df["blocker_mean_pIC50"]
    )

    pair_df.to_csv(
        outdir /
        "indole_blocker_centered_records.csv",
        index=False
    )

    # --------------------------------------------------
    # Receptor phenotypes
    # --------------------------------------------------
    receptor_response = (
        pair_df
        .groupby(
            "receptor",
            as_index=False
        )
        .agg(
            n_blockers=(
                "blocker_SMILES_std",
                "nunique"
            ),
            mean_pIC50=(
                "pIC50",
                "mean"
            ),
            median_pIC50=(
                "pIC50",
                "median"
            ),
            blocker_adjusted_response=(
                "blocker_centered_pIC50",
                "mean"
            ),
            median_blocker_adjusted_response=(
                "blocker_centered_pIC50",
                "median"
            )
        )
    )

    summary = receptor_seq.merge(
        receptor_response,
        on="receptor",
        how="inner"
    )

    summary = summary.sort_values(
        "isoleucine_fraction"
    ).reset_index(
        drop=True
    )

    summary.to_csv(
        outdir /
        "receptor_isoleucine_response_summary.csv",
        index=False
    )

    # --------------------------------------------------
    # Correlation results
    # --------------------------------------------------
    phenotypes = [
        "mean_pIC50",
        "median_pIC50",
        "blocker_adjusted_response",
        "median_blocker_adjusted_response"
    ]

    correlation_rows = []

    for phenotype in phenotypes:

        for method in [
            "pearson",
            "spearman"
        ]:

            r, p_value, n = safe_corr(
                summary[
                    "isoleucine_fraction"
                ],
                summary[
                    phenotype
                ],
                method
            )

            correlation_rows.append({
                "phenotype":
                    phenotype,
                "method":
                    method,
                "n_receptors":
                    n,
                "correlation":
                    r,
                "p_value":
                    p_value
            })

    correlations = pd.DataFrame(
        correlation_rows
    )

    correlations.to_csv(
        outdir /
        "correlation_results.csv",
        index=False
    )

    # --------------------------------------------------
    # Leave-one-receptor-out correlation sensitivity
    # --------------------------------------------------
    loro_corr = pd.concat(
        [
            leave_one_receptor_out_correlations(
                summary,
                phenotype
            )
            for phenotype in phenotypes
        ],
        ignore_index=True
    )

    loro_corr.to_csv(
        outdir /
        "leave_one_receptor_out_correlation_sensitivity.csv",
        index=False
    )

    # --------------------------------------------------
    # Figures
    # --------------------------------------------------
    figure_specs = [
        (
            "mean_pIC50",
            "Mean pIC50",
            (
                "Isoleucine composition vs "
                "mean antagonist response"
            ),
            "figure_isoleucine_vs_mean_pic50"
        ),
        (
            "median_pIC50",
            "Median pIC50",
            (
                "Isoleucine composition vs "
                "median antagonist response"
            ),
            "figure_isoleucine_vs_median_pic50"
        ),
        (
            "blocker_adjusted_response",
            (
                "Blocker-adjusted receptor "
                "response (mean centered pIC50)"
            ),
            (
                "Isoleucine composition vs "
                "blocker-adjusted response"
            ),
            "figure_isoleucine_vs_adjusted_response"
        ),
        (
            "median_blocker_adjusted_response",
            (
                "Median blocker-adjusted "
                "response"
            ),
            (
                "Isoleucine composition vs "
                "median blocker-adjusted response"
            ),
            "figure_isoleucine_vs_median_adjusted_response"
        )
    ]

    for (
        y_col,
        y_label,
        title,
        stem
    ) in figure_specs:

        for ext in [
            "png",
            "pdf"
        ]:

            make_scatter(
                summary,
                y_col,
                y_label,
                title,
                outdir /
                f"{stem}.{ext}"
            )

    # --------------------------------------------------
    # Console summary
    # --------------------------------------------------
    print(
        "=== Isoleucine composition vs "
        "Indole receptor response phenotype ==="
    )

    print(
        f"Usable records: {len(df)}"
    )

    print(
        f"Unique blockers: "
        f"{df['blocker_SMILES_std'].nunique()}"
    )

    print(
        f"Unique receptors: "
        f"{df['receptor'].nunique()}"
    )

    print()

    print(
        "=== Receptor summary ==="
    )

    print(
        summary[
            [
                "receptor",
                "sequence_length",
                "isoleucine_fraction",
                "n_blockers",
                "mean_pIC50",
                "blocker_adjusted_response"
            ]
        ].to_string(
            index=False
        )
    )

    print()

    print(
        "=== Correlations ==="
    )

    print(
        correlations.to_string(
            index=False
        )
    )

    print()

    print(
        "Blocker-adjusted response was calculated "
        "by subtracting each blocker's mean pIC50"
    )

    print(
        "across tested receptors and then averaging "
        "the centered values within each receptor."
    )

    print()

    print(
        "Use the leave-one-receptor-out sensitivity file "
        "to check whether one receptor drives the association."
    )

    print()

    print(
        "Outputs written to: "
        f"{outdir.resolve()}"
    )


if __name__ == "__main__":
    main()
