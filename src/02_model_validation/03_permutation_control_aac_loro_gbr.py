#!/usr/bin/env python3
"""
AAC + GBR receptor-level permutation control under Indole-only LORO-CV.

Observed:
    blocker ECFP4 + correct receptor amino-acid composition (AAC) -> pIC50

Null:
    randomly permute the receptor -> AAC mapping, then run exactly the same
    leave-one-receptor-out CV.

Primary statistic:
    delta RMSE = RMSE(AAC+GBR) - RMSE(antagonist-only GBR)
    More negative = larger improvement from AAC.

Also evaluates:
    number of held-out receptors for which AAC improves RMSE.

Important:
    The permutation is performed at the RECEPTOR level, not record level,
    so every record belonging to a receptor receives the same shuffled AAC.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error


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
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def clean_sequence(x):
    if pd.isna(x):
        return None
    s = "".join(str(x).split()).upper()
    if not s:
        return None
    valid = sum(c in AA_SET or c in "XBZUO" for c in s) / len(s)
    return s if valid >= 0.95 else None


def aa_composition(seq):
    counts = np.array([seq.count(aa) for aa in AA_ORDER], dtype=np.float32)
    denom = counts.sum()
    if denom <= 0:
        raise ValueError("Sequence contains no standard amino acids.")
    return counts / denom


def make_gbr(seed, n_estimators):
    return GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=0.05,
        max_depth=2,
        random_state=seed,
        loss="squared_error",
    )


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def run_loro(df, X_blocker, y, receptors, aac_map, seed, n_estimators):
    """
    Returns global OOF RMSEs and per-receptor RMSEs.
    """
    pred_ant = np.full(len(df), np.nan)
    pred_aac = np.full(len(df), np.nan)
    per_receptor = []

    receptor_array = df["receptor"].to_numpy()

    for fold_idx, heldout in enumerate(receptors, 1):
        te = np.where(receptor_array == heldout)[0]
        tr = np.where(receptor_array != heldout)[0]

        X_aac = np.stack([aac_map[r] for r in df["receptor"]]).astype(np.float32)
        X_combined = np.c_[X_blocker, X_aac]

        model_ant = make_gbr(seed + fold_idx, n_estimators)
        model_ant.fit(X_blocker[tr], y[tr])
        pa = model_ant.predict(X_blocker[te])
        pred_ant[te] = pa

        model_aac = make_gbr(seed + fold_idx, n_estimators)
        model_aac.fit(X_combined[tr], y[tr])
        ps = model_aac.predict(X_combined[te])
        pred_aac[te] = ps

        r_ant = rmse(y[te], pa)
        r_aac = rmse(y[te], ps)

        per_receptor.append({
            "heldout_receptor": heldout,
            "n_test": len(te),
            "RMSE_antagonist": r_ant,
            "RMSE_aac": r_aac,
            "delta_RMSE_aac_vs_antagonist": r_aac - r_ant,
            "aac_improved": int(r_aac < r_ant),
        })

    overall_ant = rmse(y, pred_ant)
    overall_aac = rmse(y, pred_aac)

    return {
        "RMSE_antagonist": overall_ant,
        "RMSE_aac": overall_aac,
        "delta_RMSE": overall_aac - overall_ant,
        "n_receptors_improved": int(
            sum(x["aac_improved"] for x in per_receptor)
        ),
        "per_receptor": pd.DataFrame(per_receptor),
    }


def empirical_lower_tail(observed, null):
    # Smaller/more negative is better.
    return float((1 + np.sum(null <= observed)) / (len(null) + 1))


def empirical_upper_tail(observed, null):
    # Larger is better.
    return float((1 + np.sum(null >= observed)) / (len(null) + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--sheet", default="regression_strict_exact_113")
    ap.add_argument("--outdir", default="results_indole_aac_permutation_gbr")
    ap.add_argument("--sequence_col", default="receptor_sequence")
    ap.add_argument("--n_permutations", type=int, default=1000)
    ap.add_argument("--n_estimators", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)

    if "endpoint_type" in df.columns:
        df = df[
            df["endpoint_type"].astype(str).str.upper() == "IC50"
        ].copy()

    df = df[
        df["malodor"].astype(str).str.strip().str.lower() == "indole"
    ].copy()

    activity_col = next(
        c for c in ["activity_uM", "IC50_uM", "IC50", "activity_value"]
        if c in df.columns
    )

    df["IC50_uM_numeric"] = pd.to_numeric(
        df[activity_col], errors="coerce"
    )
    df["blocker_SMILES_std"] = df["blocker_SMILES"].apply(
        standardize_smiles
    )
    df["seq"] = df[args.sequence_col].apply(clean_sequence)

    df = df.dropna(
        subset=[
            "receptor",
            "blocker_SMILES_std",
            "seq",
            "IC50_uM_numeric",
        ]
    )
    df = df[df["IC50_uM_numeric"] > 0].reset_index(drop=True)
    df["receptor"] = df["receptor"].astype(str).str.strip()
    df["pIC50"] = 6.0 - np.log10(df["IC50_uM_numeric"])

    # ECFP4
    fpgen = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048
    )
    X_blocker = np.stack([
        np.asarray(
            fpgen.GetFingerprint(Chem.MolFromSmiles(s)),
            dtype=np.float32,
        )
        for s in df["blocker_SMILES_std"]
    ])

    # One sequence per receptor
    rs = df[["receptor", "seq"]].drop_duplicates()
    bad = rs.groupby("receptor")["seq"].nunique()
    bad = bad[bad > 1]
    if len(bad):
        raise ValueError(
            "Multiple sequences found for: " + ", ".join(bad.index)
        )

    rs = rs.drop_duplicates("receptor").reset_index(drop=True)
    receptors = rs["receptor"].tolist()

    true_aac_map = {
        r: aa_composition(s)
        for r, s in zip(rs["receptor"], rs["seq"])
    }

    y = df["pIC50"].to_numpy()

    print("=== Indole-only AAC + GBR permutation control ===")
    print(f"Usable rows: {len(df)}")
    print(f"Unique blockers: {df['blocker_SMILES_std'].nunique()}")
    print(f"Unique receptors: {len(receptors)}")
    print(f"AAC dimension: 20")
    print(f"Permutations: {args.n_permutations}")
    print()

    # Observed
    observed = run_loro(
        df, X_blocker, y, receptors,
        true_aac_map, args.seed, args.n_estimators
    )

    observed["per_receptor"].to_csv(
        outdir / "observed_per_receptor.csv", index=False
    )

    print(f"Observed RMSE antagonist: {observed['RMSE_antagonist']:.6f}")
    print(f"Observed RMSE AAC:        {observed['RMSE_aac']:.6f}")
    print(f"Observed delta RMSE:      {observed['delta_RMSE']:.6f}")
    print(
        "Receptors improved:       "
        f"{observed['n_receptors_improved']}/{len(receptors)}"
    )
    print()

    # Permutation null:
    # shuffle receptor labels attached to the AAC vectors.
    rng = np.random.default_rng(args.seed)
    original_vectors = [true_aac_map[r].copy() for r in receptors]

    null_rows = []

    for perm_i in range(1, args.n_permutations + 1):
        perm = rng.permutation(len(receptors))
        shuffled_map = {
            receptor: original_vectors[perm[j]]
            for j, receptor in enumerate(receptors)
        }

        result = run_loro(
            df, X_blocker, y, receptors,
            shuffled_map,
            args.seed, args.n_estimators
        )

        null_rows.append({
            "permutation": perm_i,
            "RMSE_antagonist": result["RMSE_antagonist"],
            "RMSE_aac": result["RMSE_aac"],
            "delta_RMSE": result["delta_RMSE"],
            "n_receptors_improved": result["n_receptors_improved"],
        })

        if perm_i % 100 == 0 or perm_i == args.n_permutations:
            print(f"Permutation {perm_i}/{args.n_permutations}")

    null_df = pd.DataFrame(null_rows)
    null_df.to_csv(outdir / "permutation_results.csv", index=False)

    p_delta = empirical_lower_tail(
        observed["delta_RMSE"],
        null_df["delta_RMSE"].to_numpy()
    )
    p_aac_rmse = empirical_lower_tail(
        observed["RMSE_aac"],
        null_df["RMSE_aac"].to_numpy()
    )
    p_n_improved = empirical_upper_tail(
        observed["n_receptors_improved"],
        null_df["n_receptors_improved"].to_numpy()
    )

    summary = pd.DataFrame([{
        "estimator": "gbr",
        "n_permutations": args.n_permutations,
        "observed_RMSE_antagonist": observed["RMSE_antagonist"],
        "observed_RMSE_aac": observed["RMSE_aac"],
        "observed_delta_RMSE": observed["delta_RMSE"],
        "observed_n_receptors_improved":
            observed["n_receptors_improved"],
        "null_mean_delta_RMSE":
            null_df["delta_RMSE"].mean(),
        "null_median_delta_RMSE":
            null_df["delta_RMSE"].median(),
        "null_sd_delta_RMSE":
            null_df["delta_RMSE"].std(ddof=1),
        "empirical_p_delta_RMSE": p_delta,
        "empirical_p_RMSE_aac": p_aac_rmse,
        "empirical_p_n_receptors_improved": p_n_improved,
    }])

    summary.to_csv(
        outdir / "permutation_summary.csv", index=False
    )

    # Histogram
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(null_df["delta_RMSE"], bins=30)
    ax.axvline(
        observed["delta_RMSE"],
        linestyle="--",
        linewidth=2,
        label=f"Observed = {observed['delta_RMSE']:.3f}"
    )
    ax.set_xlabel("Delta RMSE (AAC - antagonist)")
    ax.set_ylabel("Permutation count")
    ax.set_title("AAC receptor-level permutation control (GBR)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        outdir / "figure_aac_permutation_delta_rmse.png",
        dpi=300, bbox_inches="tight"
    )
    fig.savefig(
        outdir / "figure_aac_permutation_delta_rmse.pdf",
        bbox_inches="tight"
    )
    plt.close(fig)

    # n improved histogram
    fig, ax = plt.subplots(figsize=(8, 6))
    bins = np.arange(-0.5, len(receptors) + 1.5, 1)
    ax.hist(null_df["n_receptors_improved"], bins=bins)
    ax.axvline(
        observed["n_receptors_improved"],
        linestyle="--",
        linewidth=2,
        label=(
            f"Observed = "
            f"{observed['n_receptors_improved']}/{len(receptors)}"
        )
    )
    ax.set_xlabel("Number of receptors improved by AAC")
    ax.set_ylabel("Permutation count")
    ax.set_title("AAC receptor-level permutation control (GBR)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        outdir / "figure_aac_permutation_n_improved.png",
        dpi=300, bbox_inches="tight"
    )
    fig.savefig(
        outdir / "figure_aac_permutation_n_improved.pdf",
        bbox_inches="tight"
    )
    plt.close(fig)

    print()
    print("=== Permutation-control summary ===")
    print(summary.to_string(index=False))
    print()
    print(
        "Interpretation: more negative delta RMSE means AAC provides "
        "greater improvement over antagonist-only GBR."
    )
    print(
        "The null distribution is generated by shuffling complete AAC "
        "vectors among receptor identities."
    )
    print()
    print("Outputs written to:", outdir.resolve())


if __name__ == "__main__":
    main()
