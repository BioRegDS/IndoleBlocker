#!/usr/bin/env python3
"""
Positive-control enrichment analysis for IFRA Indole virtual screening.

Input:
    positive_control_virtual_screen_ranks.csv
    ifra_compound_ranking_novel_only.csv
from 03_virtual_screen_ifra_indole_aac_gbr.py

Primary test:
Randomly sample K novel compounds (K = number of eligible unseen positive controls)
without replacement and compare their mean rank with the observed positive-control
mean rank. Lower rank = better.

Also reports:
- mean/median rank
- mean percentile
- top-X% enrichment counts
- empirical permutation p-values
- hypergeometric one-sided p-values for top-X% counts
- null-distribution figure
- positive-control rank/percentile figure

Training compounds are excluded automatically.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import hypergeom


def as_bool(x):
    if isinstance(x, bool):
        return x
    if pd.isna(x):
        return False
    return str(x).strip().lower() in {"true", "1", "yes", "y"}


def empirical_p_lower(null, observed):
    # +1 correction prevents zero p-values.
    return (np.sum(null <= observed) + 1.0) / (len(null) + 1.0)


def empirical_p_upper(null, observed):
    return (np.sum(null >= observed) + 1.0) / (len(null) + 1.0)


def percentile_from_rank(rank, n):
    # Same convention as screening script: 100 = best.
    if n <= 1:
        return 100.0
    return 100.0 * (n - rank) / (n - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", required=True)
    ap.add_argument("--n_permutations", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--top_percentages",
        default="1,5,10,20,25,40",
        help="Comma-separated top-percent thresholds."
    )
    ap.add_argument(
        "--outdir",
        default=None,
        help="Default: RESULT_DIR/positive_control_enrichment"
    )
    args = ap.parse_args()

    result_dir = Path(args.result_dir)
    outdir = Path(args.outdir) if args.outdir else (
        result_dir / "positive_control_enrichment"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    ranking_path = result_dir / "ifra_compound_ranking_novel_only.csv"
    pc_path = result_dir / "positive_control_virtual_screen_ranks.csv"

    if not ranking_path.exists():
        raise FileNotFoundError(ranking_path)
    if not pc_path.exists():
        raise FileNotFoundError(pc_path)

    ranking = pd.read_csv(ranking_path)
    pcs = pd.read_csv(pc_path)

    required_rank = [
        "SMILES_std",
        "novel_broad_spectrum_rank",
        "mean_predicted_pIC50",
    ]
    missing = [c for c in required_rank if c not in ranking.columns]
    if missing:
        raise ValueError(
            f"Novel ranking file missing columns: {missing}"
        )

    if "is_training_compound" in pcs.columns:
        pcs["is_training_compound"] = pcs[
            "is_training_compound"
        ].apply(as_bool)
        pcs = pcs[~pcs["is_training_compound"]].copy()

    pcs = pcs.dropna(
        subset=["novel_broad_spectrum_rank"]
    ).copy()

    if len(pcs) == 0:
        raise ValueError(
            "No unseen positive controls with novel ranks were found."
        )

    N = len(ranking)
    K = len(pcs)

    ranks = ranking["novel_broad_spectrum_rank"].astype(int).to_numpy()
    observed_ranks = pcs[
        "novel_broad_spectrum_rank"
    ].astype(float).to_numpy()

    if K > N:
        raise ValueError("More positive controls than library compounds.")

    observed_mean_rank = float(np.mean(observed_ranks))
    observed_median_rank = float(np.median(observed_ranks))
    observed_mean_percentile = float(
        np.mean([percentile_from_rank(r, N) for r in observed_ranks])
    )

    print("=== Positive-control enrichment analysis ===")
    print(f"Novel library compounds: {N}")
    print(f"Eligible unseen positive controls: {K}")
    print(f"Permutations: {args.n_permutations}")
    print()
    print("Positive controls:")
    show_cols = [
        c for c in [
            "positive_control_name",
            "novel_broad_spectrum_rank",
            "novel_mean_pIC50_percentile",
            "mean_predicted_pIC50",
        ]
        if c in pcs.columns
    ]
    print(pcs[show_cols].sort_values(
        "novel_broad_spectrum_rank"
    ).to_string(index=False))
    print()
    print(f"Observed mean rank:   {observed_mean_rank:.3f}")
    print(f"Observed median rank: {observed_median_rank:.3f}")
    print(f"Observed mean percentile: {observed_mean_percentile:.3f}")

    rng = np.random.default_rng(args.seed)

    # Vectorized in chunks to keep memory small even for large permutation counts.
    null_mean = np.empty(args.n_permutations, dtype=float)
    null_median = np.empty(args.n_permutations, dtype=float)

    top_percentages = [
        float(x.strip())
        for x in args.top_percentages.split(",")
        if x.strip()
    ]
    null_top_counts = {
        p: np.empty(args.n_permutations, dtype=np.int16)
        for p in top_percentages
    }

    chunk = 10000
    done = 0
    while done < args.n_permutations:
        n_now = min(chunk, args.n_permutations - done)

        # K is tiny here; row-wise choice without replacement is inexpensive.
        sampled = np.empty((n_now, K), dtype=int)
        for i in range(n_now):
            sampled[i] = rng.choice(
                ranks, size=K, replace=False
            )

        null_mean[done:done+n_now] = sampled.mean(axis=1)
        null_median[done:done+n_now] = np.median(
            sampled, axis=1
        )

        for p in top_percentages:
            cutoff = int(np.ceil(N * p / 100.0))
            null_top_counts[p][done:done+n_now] = (
                sampled <= cutoff
            ).sum(axis=1)

        done += n_now
        if done % 10000 == 0 or done == args.n_permutations:
            print(f"Permutation {done}/{args.n_permutations}")

    p_mean = empirical_p_lower(
        null_mean, observed_mean_rank
    )
    p_median = empirical_p_lower(
        null_median, observed_median_rank
    )

    summary_rows = [
        {
            "analysis": "mean_rank",
            "observed": observed_mean_rank,
            "null_mean": float(null_mean.mean()),
            "null_median": float(np.median(null_mean)),
            "empirical_p_one_sided": p_mean,
        },
        {
            "analysis": "median_rank",
            "observed": observed_median_rank,
            "null_mean": float(null_median.mean()),
            "null_median": float(np.median(null_median)),
            "empirical_p_one_sided": p_median,
        },
    ]

    threshold_rows = []
    for p in top_percentages:
        cutoff = int(np.ceil(N * p / 100.0))
        observed_count = int(
            np.sum(observed_ranks <= cutoff)
        )

        perm_p = empirical_p_upper(
            null_top_counts[p], observed_count
        )

        # Exact hypergeometric probability:
        # population N, top-bin size cutoff, draw K controls,
        # P(X >= observed_count).
        hyper_p = float(
            hypergeom.sf(
                observed_count - 1,
                N,
                cutoff,
                K
            )
        )

        threshold_rows.append({
            "top_percentage": p,
            "rank_cutoff": cutoff,
            "n_positive_controls": K,
            "observed_controls_in_top_bin": observed_count,
            "expected_controls_in_top_bin": K * cutoff / N,
            "empirical_p_one_sided": perm_p,
            "hypergeometric_p_one_sided": hyper_p,
        })

    summary = pd.DataFrame(summary_rows)
    thresholds = pd.DataFrame(threshold_rows)

    summary.to_csv(
        outdir / "rank_enrichment_summary.csv",
        index=False
    )
    thresholds.to_csv(
        outdir / "top_percentage_enrichment.csv",
        index=False
    )

    pc_export = pcs.copy()
    pc_export["recomputed_percentile"] = [
        percentile_from_rank(r, N)
        for r in pc_export["novel_broad_spectrum_rank"]
    ]
    pc_export.to_csv(
        outdir / "positive_controls_analyzed.csv",
        index=False
    )

    # Save null distributions for reproducibility / supplementary analysis.
    null_df = pd.DataFrame({
        "permutation": np.arange(
            1, args.n_permutations + 1
        ),
        "mean_rank": null_mean,
        "median_rank": null_median,
    })
    for p in top_percentages:
        label = str(p).replace(".", "p")
        null_df[f"n_in_top_{label}pct"] = null_top_counts[p]
    null_df.to_csv(
        outdir / "permutation_null_distribution.csv",
        index=False
    )

    # Figure 1: null mean-rank distribution.
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(null_mean, bins=50)
    ax.axvline(
        observed_mean_rank,
        linestyle="--",
        linewidth=2,
        label=(
            f"Observed mean rank = "
            f"{observed_mean_rank:.1f}\n"
            f"Empirical p = {p_mean:.4g}"
        ),
    )
    ax.set_xlabel(
        f"Mean rank of {K} randomly selected compounds"
    )
    ax.set_ylabel("Permutation count")
    ax.set_title(
        "Positive-control enrichment in IFRA virtual screening"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        outdir / "positive_control_mean_rank_permutation.png",
        dpi=300
    )
    fig.savefig(
        outdir / "positive_control_mean_rank_permutation.pdf"
    )
    plt.close(fig)

    # Figure 2: positive-control positions.
    pcs_plot = pc_export.sort_values(
        "novel_broad_spectrum_rank",
        ascending=True
    ).copy()

    names = (
        pcs_plot["positive_control_name"].astype(str).tolist()
        if "positive_control_name" in pcs_plot.columns
        else [f"Control {i+1}" for i in range(len(pcs_plot))]
    )
    y = np.arange(len(pcs_plot))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(
        pcs_plot["novel_broad_spectrum_rank"],
        y,
        s=70
    )
    for yi, rank in zip(
        y,
        pcs_plot["novel_broad_spectrum_rank"]
    ):
        ax.text(
            rank,
            yi + 0.12,
            f"{int(rank)}/{N}",
            ha="center",
            fontsize=9
        )

    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(0, N + 1)
    ax.invert_yaxis()
    ax.set_xlabel(
        "Rank among novel IFRA compounds (lower is better)"
    )
    ax.set_title(
        "Ranks of unseen positive controls"
    )

    # Reference boundaries.
    for pct in [10, 20, 40]:
        cutoff = N * pct / 100.0
        ax.axvline(
            cutoff,
            linestyle=":",
            linewidth=1
        )
        ax.text(
            cutoff,
            len(y) - 0.55,
            f"top {pct}%",
            rotation=90,
            va="bottom",
            ha="right",
            fontsize=8
        )

    fig.tight_layout()
    fig.savefig(
        outdir / "positive_control_ranks.png",
        dpi=300
    )
    fig.savefig(
        outdir / "positive_control_ranks.pdf"
    )
    plt.close(fig)

    metadata = {
        "novel_library_size": N,
        "n_unseen_positive_controls": K,
        "n_permutations": args.n_permutations,
        "seed": args.seed,
        "primary_test":
            "one-sided empirical permutation test of mean rank; "
            "lower mean rank indicates enrichment near the top",
        "sampling":
            "K compounds sampled uniformly without replacement from "
            "the novel-only IFRA ranking",
        "training_compounds_excluded": True,
        "rank_direction": "1 = best/highest mean predicted pIC50",
    }
    (outdir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8"
    )

    print("\n=== Rank enrichment ===")
    print(summary.to_string(index=False))

    print("\n=== Top-percentage enrichment ===")
    print(thresholds.to_string(index=False))

    print(
        "\nInterpretation: small one-sided p-values mean that "
        "the unseen positive controls are ranked higher than expected "
        "for randomly selected novel IFRA compounds."
    )
    print(f"\nOutputs written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
