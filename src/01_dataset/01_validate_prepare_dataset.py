#!/usr/bin/env python3
"""
Validate Supplementary Table S1 and prepare analysis-ready datasets.

Starting input
--------------
data/Supplementary_Table_S1_curated_patent_OR_antagonist_dataset.csv

Outputs
-------
results/dataset/curated_activity_dataset.csv
results/dataset/dataset_summary.csv
results/dataset/indole_ic50_dataset.csv
results/dataset/indole_ic50_summary.csv
results/dataset/indole_pic50_matrix.csv

The script intentionally starts from the manually curated Supplementary Table S1.
It checks pActivity values, standardizes blocker SMILES with RDKit, and extracts
the 37-record Indole IC50 dataset used for receptor-aware modeling.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


def standardize_smiles(smiles):
    if pd.isna(smiles):
        return None
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        mol = rdMolStandardize.FragmentParent(mol)
        mol = rdMolStandardize.Normalizer().normalize(mol)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="data/Supplementary_Table_S1_curated_patent_OR_antagonist_dataset.csv",
    )
    ap.add_argument("--outdir", default="results/dataset")
    ap.add_argument("--tolerance", type=float, default=1e-8)
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    d = pd.read_csv(args.input)

    required = [
        "record_id", "source_patent", "malodor", "receptor", "species",
        "blocker", "blocker_SMILES", "endpoint_type", "activity_uM",
        "pActivity_M", "activity_quality", "source_location",
        "structure_confidence",
    ]
    missing = [c for c in required if c not in d.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if d["record_id"].duplicated().any():
        dup = d.loc[d["record_id"].duplicated(), "record_id"].tolist()
        raise ValueError(f"Duplicate record_id values: {dup}")

    d["activity_uM"] = pd.to_numeric(d["activity_uM"], errors="coerce")
    d["pActivity_M"] = pd.to_numeric(d["pActivity_M"], errors="coerce")

    if d["activity_uM"].isna().any() or (d["activity_uM"] <= 0).any():
        raise ValueError("All activity_uM values must be positive numeric values.")

    # Recalculate pActivity = -log10(activity in mol/L) = 6 - log10(uM)
    d["pActivity_M_recalculated"] = 6.0 - np.log10(d["activity_uM"])
    d["pActivity_abs_diff"] = (
        d["pActivity_M"] - d["pActivity_M_recalculated"]
    ).abs()

    max_diff = float(d["pActivity_abs_diff"].max())
    if max_diff > args.tolerance:
        raise ValueError(
            f"pActivity validation failed: max absolute difference = {max_diff:.3g}"
        )

    d["blocker_SMILES_std"] = d["blocker_SMILES"].apply(standardize_smiles)
    n_bad_smiles = int(d["blocker_SMILES_std"].isna().sum())
    if n_bad_smiles:
        raise ValueError(f"RDKit standardization failed for {n_bad_smiles} blocker SMILES.")

    d.to_csv(out / "curated_activity_dataset.csv", index=False)

    summary_rows = []
    summary_rows.append({"metric": "n_records", "value": len(d)})
    summary_rows.append({"metric": "n_source_patents", "value": d["source_patent"].nunique()})
    summary_rows.append({"metric": "n_malodors", "value": d["malodor"].nunique()})
    summary_rows.append({"metric": "n_unique_blockers", "value": d["blocker_SMILES_std"].nunique()})
    summary_rows.append({"metric": "n_IC50", "value": int((d["endpoint_type"] == "IC50").sum())})
    summary_rows.append({"metric": "n_EC50", "value": int((d["endpoint_type"] == "EC50").sum())})
    summary_rows.append({"metric": "max_pActivity_abs_diff", "value": max_diff})
    pd.DataFrame(summary_rows).to_csv(out / "dataset_summary.csv", index=False)

    # Primary dataset used in Results 2-5.
    indole = d[
        (d["malodor"].astype(str).str.lower() == "indole")
        & (d["endpoint_type"].astype(str).str.upper() == "IC50")
    ].copy()

    indole.to_csv(out / "indole_ic50_dataset.csv", index=False)

    indole_summary = pd.DataFrame([
        {"metric": "n_records", "value": len(indole)},
        {"metric": "n_unique_blockers", "value": indole["blocker_SMILES_std"].nunique()},
        {"metric": "n_unique_receptors", "value": indole["receptor"].nunique()},
        {"metric": "min_pIC50", "value": indole["pActivity_M_recalculated"].min()},
        {"metric": "median_pIC50", "value": indole["pActivity_M_recalculated"].median()},
        {"metric": "max_pIC50", "value": indole["pActivity_M_recalculated"].max()},
    ])
    indole_summary.to_csv(out / "indole_ic50_summary.csv", index=False)

    # Matrix used for Figure 1C: blocker x receptor, pIC50 values.
    matrix = indole.pivot_table(
        index="blocker",
        columns="receptor",
        values="pActivity_M_recalculated",
        aggfunc="mean",
    )
    matrix.to_csv(out / "indole_pic50_matrix.csv")

    print("=== Dataset validation/preparation ===")
    print(f"Input records: {len(d)}")
    print(f"Source patents: {d['source_patent'].nunique()}")
    print(f"IC50 / EC50: {(d['endpoint_type']=='IC50').sum()} / {(d['endpoint_type']=='EC50').sum()}")
    print(f"Max pActivity difference: {max_diff:.3g}")
    print()
    print("=== Primary Indole IC50 dataset ===")
    print(f"Records: {len(indole)}")
    print(f"Unique blockers: {indole['blocker_SMILES_std'].nunique()}")
    print(f"Unique receptors: {indole['receptor'].nunique()}")
    print(f"Outputs written to: {out.resolve()}")


if __name__ == "__main__":
    main()
