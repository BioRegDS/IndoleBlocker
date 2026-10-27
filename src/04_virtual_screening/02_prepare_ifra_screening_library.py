#!/usr/bin/env python3
"""
Prepare an IFRA PubChem library for OdorBlock virtual screening.

Key design choice:
Chemical standardization is intentionally matched to the training pipeline:
    MolFromSmiles -> FragmentParent -> Normalizer -> canonical isomeric SMILES

Outputs
-------
ifra_screening_library.csv
ifra_screening_library_all_records.csv
ifra_excluded_records.csv
ifra_duplicate_records.csv
training_compound_overlap.csv
positive_control_check.csv
curation_summary.csv

Positive controls are identified primarily by standardized structure, not name.
The built-in structures are resolved by RDKit from the supplied SMILES.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize


# Screening controls discussed for this project.
# alpha-Ionone is the primary Indole-related retrospective control.
POSITIVE_CONTROLS = {
    "alpha-Ionone": {
        "smiles": "CC1=CCCC(C1/C=C/C(=O)C)(C)C",
        "type": "primary_indole_control",
    },
    "Lilial": {
        "smiles": "CC(CC1=CC=C(C=C1)C(C)(C)C)C=O",
        "type": "general_malodor_control",
    },
    "Bourgeonal": {
        "smiles": "CC(C)(C)C1=CC=C(C=C1)CCC=O",
        "type": "general_malodor_control",
    },
    "Helional": {
        "smiles": "CC(CC1=CC2=C(C=C1)OCO2)C=O",
        "type": "general_malodor_control",
    },
    "Cinnamaldehyde": {
        "smiles": "C1=CC=C(C=C1)/C=C/C=O",
        "type": "general_malodor_control",
    },
    "Citral": {
        "smiles": "O=CC=C(C)CCC=C(C)C",
        "type": "general_malodor_control",
    },
    "Cashmeran": {
        "smiles": "CC1C(C)(C)C2=C(C(=O)CCC2)C1(C)C",
        "type": "training_sanity_control",
    },
}


def standardize_smiles(smiles):
    """Same standardization logic used in the OdorBlock training script."""
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


def ecfp_ok(smiles, radius=2, nbits=2048):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=nbits
        )
        gen.GetFingerprint(mol)
        return True
    except Exception:
        return False


def molecular_weight(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return float(Descriptors.MolWt(mol)) if mol is not None else None
    except Exception:
        return None


def detect_smiles_column(df):
    for c in ["smiles", "connectivity_smiles", "SMILES", "canonical_smiles"]:
        if c in df.columns:
            return c
    raise ValueError(
        "No SMILES column found. Expected one of: "
        "smiles, connectivity_smiles, SMILES, canonical_smiles"
    )


def detect_training_smiles_column(df):
    for c in [
        "blocker_SMILES_std",
        "blocker_SMILES",
        "blocker_smiles_std",
        "blocker_smiles",
        "SMILES",
        "smiles",
    ]:
        if c in df.columns:
            return c
    raise ValueError("Could not identify the blocker SMILES column in training data.")


def load_training(path, sheet):
    if path is None:
        return None, set()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix.lower() in [".xlsx", ".xls"]:
        d = pd.read_excel(p, sheet_name=sheet)
    else:
        d = pd.read_csv(p)
    col = detect_training_smiles_column(d)
    d = d.copy()
    d["training_SMILES_std"] = d[col].apply(standardize_smiles)
    vals = set(d["training_SMILES_std"].dropna().unique())
    return d, vals


def build_control_table():
    rows = []
    for name, info in POSITIVE_CONTROLS.items():
        std = standardize_smiles(info["smiles"])
        rows.append({
            "positive_control_name": name,
            "positive_control_type": info["type"],
            "reference_SMILES": info["smiles"],
            "reference_SMILES_std": std,
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="ifra_pubchem_raw.csv from downloader v3")
    ap.add_argument("--training", default=None,
                    help="Optional training XLSX/CSV for blocker-overlap detection")
    ap.add_argument("--training_sheet", default="regression_strict_exact_113")
    ap.add_argument("--outdir", default="data/ifra_screening")
    ap.add_argument("--ecfp_radius", type=int, default=2)
    ap.add_argument("--ecfp_bits", type=int, default=2048)
    ap.add_argument("--min_mw", type=float, default=0.0,
                    help="Optional MW filter; default does not exclude by MW")
    ap.add_argument("--max_mw", type=float, default=100000.0,
                    help="Optional MW filter; default effectively disabled")
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    n_input = len(df)
    smiles_col = detect_smiles_column(df)

    print("=== Prepare IFRA virtual-screening library ===")
    print(f"Input records: {n_input}")
    print(f"SMILES column: {smiles_col}")
    print("Standardization: FragmentParent -> Normalizer -> canonical isomeric SMILES")
    print()

    d = df.copy()
    d["input_SMILES"] = d[smiles_col]
    d["n_input_fragments"] = d["input_SMILES"].fillna("").astype(str).str.count(r"\.") + 1
    d.loc[d["input_SMILES"].isna(), "n_input_fragments"] = 0
    d["was_multifragment_input"] = d["n_input_fragments"] > 1

    # Same chemical representation as training.
    d["SMILES_std"] = d["input_SMILES"].apply(standardize_smiles)
    d["rdkit_valid"] = d["SMILES_std"].notna()
    d["MW_rdkit_std"] = d["SMILES_std"].apply(
        lambda x: molecular_weight(x) if pd.notna(x) else None
    )
    d["ecfp4_valid"] = d["SMILES_std"].apply(
        lambda x: ecfp_ok(x, args.ecfp_radius, args.ecfp_bits)
        if pd.notna(x) else False
    )

    # Do not automatically discard a salt merely because the input had dots:
    # FragmentParent is the same behavior used in training. We retain a flag.
    d["passes_mw_filter"] = (
        d["MW_rdkit_std"].notna()
        & (d["MW_rdkit_std"] >= args.min_mw)
        & (d["MW_rdkit_std"] <= args.max_mw)
    )
    d["curation_eligible"] = (
        d["rdkit_valid"] & d["ecfp4_valid"] & d["passes_mw_filter"]
    )

    reasons = []
    for _, r in d.iterrows():
        rr = []
        if not r["rdkit_valid"]:
            rr.append("rdkit_standardization_failed")
        elif not r["ecfp4_valid"]:
            rr.append("ecfp4_failed")
        if r["rdkit_valid"] and not r["passes_mw_filter"]:
            rr.append("outside_mw_filter")
        reasons.append(";".join(rr))
    d["exclusion_reason"] = reasons

    eligible = d[d["curation_eligible"]].copy()

    # Structure-based training overlap.
    training_df, training_set = load_training(args.training, args.training_sheet)
    eligible["is_training_compound"] = eligible["SMILES_std"].isin(training_set)

    # Structure-based positive controls.
    controls = build_control_table()
    control_map = {}
    for _, r in controls.iterrows():
        if pd.notna(r["reference_SMILES_std"]):
            control_map.setdefault(r["reference_SMILES_std"], []).append(
                (r["positive_control_name"], r["positive_control_type"])
            )

    pc_names, pc_types = [], []
    for smi in eligible["SMILES_std"]:
        hits = control_map.get(smi, [])
        pc_names.append(";".join(x[0] for x in hits))
        pc_types.append(";".join(x[1] for x in hits))
    eligible["positive_control_name"] = pc_names
    eligible["positive_control_type"] = pc_types
    eligible["is_positive_control"] = eligible["positive_control_name"].ne("")

    # Duplicate standardized structures: keep one representative in final library.
    eligible["duplicate_count"] = eligible.groupby("SMILES_std")["SMILES_std"].transform("size")
    sort_cols = ["SMILES_std"]
    if "pubchem_cid" in eligible.columns:
        sort_cols.append("pubchem_cid")
    eligible = eligible.sort_values(sort_cols).reset_index(drop=True)
    eligible["duplicate_rank"] = eligible.groupby("SMILES_std").cumcount() + 1

    duplicates = eligible[eligible["duplicate_count"] > 1].copy()
    final = eligible[eligible["duplicate_rank"] == 1].copy()

    # Useful screening flags. Training compounds remain in the file for sanity checks,
    # but should be excluded from novel-hit ranking downstream.
    final["eligible_for_novel_hit_ranking"] = ~final["is_training_compound"]

    # Save.
    d.to_csv(out / "ifra_screening_library_all_records.csv", index=False)
    d[~d["curation_eligible"]].to_csv(out / "ifra_excluded_records.csv", index=False)
    duplicates.to_csv(out / "ifra_duplicate_records.csv", index=False)
    final.to_csv(out / "ifra_screening_library.csv", index=False)
    final[final["is_training_compound"]].to_csv(
        out / "training_compound_overlap.csv", index=False
    )

    # Positive-control report includes missing controls too.
    pc_report = []
    for _, c in controls.iterrows():
        hit = final[final["SMILES_std"] == c["reference_SMILES_std"]]
        if len(hit):
            for _, h in hit.iterrows():
                pc_report.append({
                    "positive_control_name": c["positive_control_name"],
                    "positive_control_type": c["positive_control_type"],
                    "found_in_ifra_library": True,
                    "is_training_compound": bool(h["is_training_compound"]),
                    "pubchem_cid": h.get("pubchem_cid", None),
                    "pubchem_title": h.get("pubchem_title", None),
                    "SMILES_std": h["SMILES_std"],
                })
        else:
            pc_report.append({
                "positive_control_name": c["positive_control_name"],
                "positive_control_type": c["positive_control_type"],
                "found_in_ifra_library": False,
                "is_training_compound": False,
                "pubchem_cid": None,
                "pubchem_title": None,
                "SMILES_std": c["reference_SMILES_std"],
            })
    pc_report = pd.DataFrame(pc_report)
    pc_report.to_csv(out / "positive_control_check.csv", index=False)

    summary = pd.DataFrame([
        ["input_records", n_input],
        ["rdkit_valid", int(d["rdkit_valid"].sum())],
        ["ecfp4_valid", int(d["ecfp4_valid"].sum())],
        ["multifragment_inputs", int(d["was_multifragment_input"].sum())],
        ["curation_eligible_records", len(eligible)],
        ["unique_standardized_structures", len(final)],
        ["duplicate_records_beyond_first", int(len(eligible) - len(final))],
        ["training_compounds_in_final_library", int(final["is_training_compound"].sum())],
        ["positive_controls_found", int(pc_report["found_in_ifra_library"].sum())],
        ["novel_ranking_candidates", int(final["eligible_for_novel_hit_ranking"].sum())],
    ], columns=["metric", "value"])
    summary.to_csv(out / "curation_summary.csv", index=False)

    metadata = {
        "input": str(args.input),
        "training": str(args.training) if args.training else None,
        "training_sheet": args.training_sheet if args.training else None,
        "standardization":
            "RDKit MolFromSmiles -> FragmentParent -> Normalizer -> "
            "MolToSmiles(canonical=True,isomericSmiles=True)",
        "ecfp_radius": args.ecfp_radius,
        "ecfp_bits": args.ecfp_bits,
        "note":
            "Multi-fragment inputs are flagged, not automatically excluded, because "
            "FragmentParent matches the model-training standardization pipeline."
    }
    (out / "curation_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("=== Curation summary ===")
    print(summary.to_string(index=False))
    print()
    print("=== Positive-control check ===")
    cols = ["positive_control_name", "found_in_ifra_library",
            "is_training_compound", "pubchem_cid", "pubchem_title"]
    print(pc_report[cols].to_string(index=False))
    print()
    print(f"Final screening library: {out / 'ifra_screening_library.csv'}")
    print(f"Outputs written to: {out.resolve()}")


if __name__ == "__main__":
    main()
