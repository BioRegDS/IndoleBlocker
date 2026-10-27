#!/usr/bin/env python3
"""
Virtual screen the curated IFRA library against the 10 Indole receptors using
the validated OdorBlock model:

    antagonist ECFP4 (radius=2, 2048 bits) + receptor AAC (20 dims)
        -> GradientBoostingRegressor -> predicted pIC50

The final model is fitted to all usable Indole IC50 records, then predicts every
curated IFRA compound x every receptor.

Outputs include:
- all compound-receptor predictions
- compound-level broad-spectrum ranking
- novel-only ranking (training compounds excluded)
- positive-control ranks/percentiles
- training-compound predictions
- receptor-level summaries
- fitted model bundle

Important:
This is a virtual-screening prioritization model. Ranking is the primary output.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize

from sklearn.ensemble import GradientBoostingRegressor


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
            mol, canonical=True, isomericSmiles=True
        )
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
    counts = np.array(
        [seq.count(aa) for aa in AA_ORDER],
        dtype=np.float32
    )
    denom = counts.sum()
    if denom <= 0:
        raise ValueError("Sequence contains no standard amino acids.")
    return counts / denom


def make_fp(smiles, fpgen):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid standardized SMILES: {smiles}")
    return np.asarray(fpgen.GetFingerprint(mol), dtype=np.float32)


def find_activity_column(df):
    for c in ["activity_uM", "IC50_uM", "IC50", "activity_value"]:
        if c in df.columns:
            return c
    raise ValueError("Could not identify IC50/activity column.")


def load_training(args):
    df = pd.read_csv(str(args.input))

    if "endpoint_type" in df.columns:
        df = df[
            df["endpoint_type"].astype(str).str.upper() == "IC50"
        ].copy()

    if "malodor" not in df.columns:
        raise ValueError("Training data has no 'malodor' column.")

    df = df[
        df["malodor"].astype(str).str.strip().str.lower() == "indole"
    ].copy()

    activity_col = find_activity_column(df)

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
    ).copy()

    df = df[df["IC50_uM_numeric"] > 0].reset_index(drop=True)
    df["receptor"] = df["receptor"].astype(str).str.strip()
    df["pIC50"] = 6.0 - np.log10(df["IC50_uM_numeric"])

    # Require exactly one sequence per receptor.
    rs = df[["receptor", "seq"]].drop_duplicates()
    bad = rs.groupby("receptor")["seq"].nunique()
    bad = bad[bad > 1]
    if len(bad):
        raise ValueError(
            "Multiple sequences found for: " + ", ".join(bad.index)
        )

    rs = rs.drop_duplicates("receptor").reset_index(drop=True)
    return df, rs


def load_library(path):
    lib = pd.read_csv(path)

    if "SMILES_std" not in lib.columns:
        # Allow raw-ish input, but preserve the same standardization.
        for c in ["smiles", "connectivity_smiles", "SMILES"]:
            if c in lib.columns:
                lib["SMILES_std"] = lib[c].apply(standardize_smiles)
                break
        else:
            raise ValueError(
                "Library needs SMILES_std or a recognizable SMILES column."
            )

    lib = lib.dropna(subset=["SMILES_std"]).copy()

    # Final safety check: one row per standardized structure.
    if lib["SMILES_std"].duplicated().any():
        print(
            "Warning: duplicate standardized structures found in screening "
            "library; keeping first representative."
        )
        lib = lib.drop_duplicates("SMILES_std", keep="first").copy()

    if "is_training_compound" not in lib.columns:
        lib["is_training_compound"] = False

    def to_bool(x):
        if isinstance(x, bool):
            return x
        return str(x).strip().lower() in {"true", "1", "yes", "y"}

    lib["is_training_compound"] = lib["is_training_compound"].apply(to_bool)

    if "is_positive_control" not in lib.columns:
        lib["is_positive_control"] = False
    else:
        lib["is_positive_control"] = lib["is_positive_control"].apply(to_bool)

    if "positive_control_name" not in lib.columns:
        lib["positive_control_name"] = ""

    return lib.reset_index(drop=True)


def add_ranking_columns(rank_df, prefix=""):
    """Rank high predicted pIC50 as better."""
    n = len(rank_df)

    for metric in [
        "mean_predicted_pIC50",
        "median_predicted_pIC50",
        "max_predicted_pIC50",
        "min_predicted_pIC50",
    ]:
        rank_col = prefix + metric.replace("predicted_", "") + "_rank"
        pct_col = prefix + metric.replace("predicted_", "") + "_percentile"
        rank_df[rank_col] = (
            rank_df[metric]
            .rank(method="min", ascending=False)
            .astype(int)
        )
        # 100 = best, ~0 = worst.
        if n > 1:
            rank_df[pct_col] = (
                100.0 * (n - rank_df[rank_col]) / (n - 1)
            )
        else:
            rank_df[pct_col] = 100.0

    return rank_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input", required=True,
        help="Training XLSX with Indole IC50 records and receptor sequences"
    )
    ap.add_argument(
        "--library", required=True,
        help="Curated ifra_screening_library.csv"
    )
    ap.add_argument(
        "--sheet", default="regression_strict_exact_113"
    )
    ap.add_argument(
        "--sequence_col", default="receptor_sequence"
    )
    ap.add_argument(
        "--outdir", default="results_ifra_indole_aac_gbr"
    )
    ap.add_argument("--n_estimators", type=int, default=500)
    ap.add_argument("--learning_rate", type=float, default=0.05)
    ap.add_argument("--max_depth", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ecfp_radius", type=int, default=2)
    ap.add_argument("--ecfp_bits", type=int, default=2048)

    # Threshold is optional and affects only count-based summary columns.
    # 4.0 corresponds to predicted IC50 <= 100 uM.
    ap.add_argument(
        "--pic50_threshold", type=float, default=4.0,
        help="Threshold for receptor-hit count; default 4.0 (=100 uM)"
    )
    ap.add_argument(
        "--top_n", type=int, default=100,
        help="Number of top novel compounds to export separately"
    )

    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    print("=== IFRA virtual screening: Indole AAC + GBR ===")

    train, receptor_table = load_training(args)
    lib = load_library(args.library)

    receptors = receptor_table["receptor"].tolist()
    aac_map = {
        r: aa_composition(s)
        for r, s in zip(
            receptor_table["receptor"],
            receptor_table["seq"]
        )
    }

    fpgen = rdFingerprintGenerator.GetMorganGenerator(
        radius=args.ecfp_radius,
        fpSize=args.ecfp_bits
    )

    # Training matrix: exact same ECFP4 + AAC representation used in
    # the receptor representation control.
    Xb_train = np.stack([
        make_fp(s, fpgen)
        for s in train["blocker_SMILES_std"]
    ])

    Xaac_train = np.stack([
        aac_map[r] for r in train["receptor"]
    ]).astype(np.float32)

    X_train = np.c_[Xb_train, Xaac_train]
    y = train["pIC50"].to_numpy(dtype=float)

    model = GradientBoostingRegressor(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        random_state=args.seed,
        loss="squared_error",
    )
    model.fit(X_train, y)

    print(f"Training rows: {len(train)}")
    print(f"Unique training blockers: {train['blocker_SMILES_std'].nunique()}")
    print(f"Receptors: {len(receptors)} ({', '.join(receptors)})")
    print(f"Training features: {X_train.shape[1]} "
          f"({args.ecfp_bits} ECFP4 + 20 AAC)")
    print(f"IFRA compounds: {len(lib)}")
    print(f"Training compounds in library: {int(lib['is_training_compound'].sum())}")
    print(f"Novel ranking compounds: {int((~lib['is_training_compound']).sum())}")
    print()

    # Fingerprints only once per library compound.
    Xb_lib = np.stack([
        make_fp(s, fpgen)
        for s in lib["SMILES_std"]
    ])

    # Predict compound x receptor matrix.
    pred_rows = []
    for i, receptor in enumerate(receptors, 1):
        aac = aac_map[receptor]
        Xaac = np.repeat(
            aac.reshape(1, -1),
            len(lib),
            axis=0
        )
        X = np.c_[Xb_lib, Xaac]
        pred = model.predict(X)

        tmp = lib.copy()
        tmp["receptor"] = receptor
        tmp["predicted_pIC50"] = pred
        tmp["predicted_IC50_uM"] = 10.0 ** (6.0 - pred)
        tmp["above_pic50_threshold"] = (
            tmp["predicted_pIC50"] >= args.pic50_threshold
        )
        pred_rows.append(tmp)

        print(
            f"Predicted receptor {i}/{len(receptors)}: {receptor}"
        )

    pair = pd.concat(pred_rows, ignore_index=True)
    pair.to_csv(
        out / "ifra_compound_receptor_predictions.csv",
        index=False
    )

    # Compound-level aggregation.
    agg = (
        pair.groupby("SMILES_std", as_index=False)
        .agg(
            mean_predicted_pIC50=("predicted_pIC50", "mean"),
            median_predicted_pIC50=("predicted_pIC50", "median"),
            max_predicted_pIC50=("predicted_pIC50", "max"),
            min_predicted_pIC50=("predicted_pIC50", "min"),
            sd_predicted_pIC50=("predicted_pIC50", "std"),
            n_receptors_predicted=("receptor", "nunique"),
            n_receptors_above_threshold=(
                "above_pic50_threshold", "sum"
            ),
        )
    )

    # Preserve useful compound metadata from library.
    meta_cols = [
        c for c in [
            "SMILES_std",
            "pubchem_cid",
            "pubchem_title",
            "molecular_formula",
            "molecular_weight",
            "MW_rdkit_std",
            "inchikey",
            "is_training_compound",
            "eligible_for_novel_hit_ranking",
            "is_positive_control",
            "positive_control_name",
            "positive_control_type",
        ]
        if c in lib.columns
    ]
    meta = lib[meta_cols].drop_duplicates("SMILES_std")
    ranking = agg.merge(meta, on="SMILES_std", how="left")

    # Broad-spectrum score:
    # primary = mean predicted pIC50 across all 10 receptors.
    ranking = ranking.sort_values(
        ["mean_predicted_pIC50", "min_predicted_pIC50"],
        ascending=[False, False]
    ).reset_index(drop=True)

    ranking = add_ranking_columns(ranking)
    ranking["overall_broad_spectrum_rank"] = np.arange(
        1, len(ranking) + 1
    )

    ranking.to_csv(
        out / "ifra_compound_ranking_all.csv",
        index=False
    )

    # Novel-only ranking excludes all compounds overlapping the training set.
    novel = ranking[
        ~ranking["is_training_compound"].fillna(False)
    ].copy()

    novel = novel.sort_values(
        ["mean_predicted_pIC50", "min_predicted_pIC50"],
        ascending=[False, False]
    ).reset_index(drop=True)

    novel = add_ranking_columns(novel, prefix="novel_")
    novel["novel_broad_spectrum_rank"] = np.arange(
        1, len(novel) + 1
    )

    novel.to_csv(
        out / "ifra_compound_ranking_novel_only.csv",
        index=False
    )
    novel.head(args.top_n).to_csv(
        out / f"ifra_top{args.top_n}_novel_candidates.csv",
        index=False
    )

    # Positive controls: rank in all compounds and, when truly novel,
    # rank within novel-only candidates.
    pcs = ranking[
        ranking["is_positive_control"].fillna(False)
    ].copy()

    if len(pcs):
        novel_lookup = novel.set_index("SMILES_std")
        pc_rows = []

        for _, row in pcs.iterrows():
            rec = row.to_dict()
            smi = row["SMILES_std"]

            if smi in novel_lookup.index:
                nr = novel_lookup.loc[smi]
                rec["novel_broad_spectrum_rank"] = int(
                    nr["novel_broad_spectrum_rank"]
                )
                rec["novel_mean_pIC50_percentile"] = float(
                    nr["novel_mean_pIC50_percentile"]
                )
                rec["novel_candidate_count"] = len(novel)
            else:
                rec["novel_broad_spectrum_rank"] = np.nan
                rec["novel_mean_pIC50_percentile"] = np.nan
                rec["novel_candidate_count"] = len(novel)

            pc_rows.append(rec)

        pc = pd.DataFrame(pc_rows)
        pc.to_csv(
            out / "positive_control_virtual_screen_ranks.csv",
            index=False
        )
    else:
        pd.DataFrame().to_csv(
            out / "positive_control_virtual_screen_ranks.csv",
            index=False
        )

    # Training compounds retained as an internal sanity-check table.
    training_hits = ranking[
        ranking["is_training_compound"].fillna(False)
    ].copy()
    training_hits.to_csv(
        out / "training_compound_screen_predictions.csv",
        index=False
    )

    # Receptor-level prediction distribution.
    receptor_summary = (
        pair.groupby("receptor", as_index=False)
        .agg(
            mean_predicted_pIC50=("predicted_pIC50", "mean"),
            median_predicted_pIC50=("predicted_pIC50", "median"),
            min_predicted_pIC50=("predicted_pIC50", "min"),
            max_predicted_pIC50=("predicted_pIC50", "max"),
            n_compounds=("SMILES_std", "nunique"),
            n_above_threshold=("above_pic50_threshold", "sum"),
        )
    )
    receptor_summary.to_csv(
        out / "receptor_screening_summary.csv",
        index=False
    )

    # Save receptor AAC values for reproducibility.
    aac_df = pd.DataFrame([
        {
            "receptor": r,
            **{f"AAC_{aa}": float(aac_map[r][j])
               for j, aa in enumerate(AA_ORDER)}
        }
        for r in receptors
    ])
    aac_df.to_csv(
        out / "receptor_aac_features.csv",
        index=False
    )

    # Save exact training population.
    train.to_csv(
        out / "final_model_training_records.csv",
        index=False
    )

    # Model bundle for future prediction.
    joblib.dump(
        {
            "model": model,
            "feature_mode": "antagonist_receptor_aac",
            "estimator": "GradientBoostingRegressor",
            "ecfp_radius": args.ecfp_radius,
            "ecfp_bits": args.ecfp_bits,
            "aa_order": AA_ORDER,
            "receptor_aac": aac_map,
            "receptors": receptors,
            "pic50_definition": "6 - log10(IC50_uM)",
            "n_training_records": len(train),
            "n_unique_training_blockers":
                int(train["blocker_SMILES_std"].nunique()),
            "gbr_parameters": {
                "n_estimators": args.n_estimators,
                "learning_rate": args.learning_rate,
                "max_depth": args.max_depth,
                "random_state": args.seed,
                "loss": "squared_error",
            },
        },
        out / "final_indole_aac_gbr_model.joblib"
    )

    summary = {
        "training_rows": len(train),
        "unique_training_blockers":
            int(train["blocker_SMILES_std"].nunique()),
        "receptors": len(receptors),
        "library_compounds": len(lib),
        "compound_receptor_predictions": len(pair),
        "training_compounds_in_library":
            int(lib["is_training_compound"].sum()),
        "novel_ranking_compounds": len(novel),
        "ranking_primary_metric":
            "mean predicted pIC50 across all receptors",
        "ranking_tiebreaker":
            "minimum predicted pIC50 across receptors",
        "pic50_threshold": args.pic50_threshold,
        "threshold_equivalent_IC50_uM":
            float(10.0 ** (6.0 - args.pic50_threshold)),
    }
    (out / "virtual_screening_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    print("\n=== Virtual-screening summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print("\n=== Top 20 novel broad-spectrum candidates ===")
    show_cols = [
        c for c in [
            "novel_broad_spectrum_rank",
            "pubchem_cid",
            "pubchem_title",
            "mean_predicted_pIC50",
            "min_predicted_pIC50",
            "max_predicted_pIC50",
            "n_receptors_above_threshold",
            "positive_control_name",
        ]
        if c in novel.columns
    ]
    print(novel[show_cols].head(20).to_string(index=False))

    pc_path = out / "positive_control_virtual_screen_ranks.csv"
    if len(pcs):
        print("\n=== Positive controls ===")
        pc_show = pd.read_csv(pc_path)
        cols = [
            c for c in [
                "positive_control_name",
                "is_training_compound",
                "overall_broad_spectrum_rank",
                "novel_broad_spectrum_rank",
                "mean_predicted_pIC50",
                "min_predicted_pIC50",
                "n_receptors_above_threshold",
                "novel_mean_pIC50_percentile",
            ]
            if c in pc_show.columns
        ]
        print(pc_show[cols].to_string(index=False))

    print(f"\nOutputs written to: {out.resolve()}")


if __name__ == "__main__":
    main()
