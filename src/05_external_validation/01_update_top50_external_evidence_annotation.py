#!/usr/bin/env python3
"""
Update the Top-50 external-evidence annotation against the current
virtual-screening ranking.

This script:
1. reads the current novel-compound ranking;
2. selects the current Top 50 by novel_broad_spectrum_rank;
3. joins previously curated literature/patent annotations by PubChem CID;
4. verifies that every current Top-50 compound has an annotation;
5. writes the publication-ready current Top-50 annotation CSV.

The manual evidence grades are not recomputed here. This script only
synchronizes the manually curated annotation table with the current ranking.
"""

import argparse
from pathlib import Path
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ranking",
        required=True,
        help="Current ifra_compound_ranking_novel_only.csv",
    )
    p.add_argument(
        "--annotation",
        required=True,
        help="Previously curated Top-50 evidence annotation CSV",
    )
    p.add_argument(
        "--outdir",
        default="results/external_validation",
    )
    p.add_argument(
        "--top_n",
        type=int,
        default=50,
    )
    return p.parse_args()


def main():
    args = parse_args()

    ranking = pd.read_csv(args.ranking)
    annotation = pd.read_csv(args.annotation)

    required_ranking = {
        "novel_broad_spectrum_rank",
        "pubchem_cid",
        "pubchem_title",
        "mean_predicted_pIC50",
        "min_predicted_pIC50",
    }
    required_annotation = {
        "pubchem_cid",
        "grade",
        "evidence_summary",
        "key_source",
        "annotation_note",
    }

    missing_r = required_ranking - set(ranking.columns)
    missing_a = required_annotation - set(annotation.columns)

    if missing_r:
        raise ValueError(f"Ranking file missing columns: {sorted(missing_r)}")
    if missing_a:
        raise ValueError(f"Annotation file missing columns: {sorted(missing_a)}")

    top = (
        ranking.sort_values("novel_broad_spectrum_rank")
        .head(args.top_n)
        .copy()
    )

    # Keep only manually curated evidence fields from the annotation.
    # All ranking-dependent fields are taken from the current ranking CSV.
    evidence_columns = [
        "pubchem_cid",
        "grade",
        "evidence_summary",
        "key_source",
        "annotation_note",
    ]
    
    missing_evidence = set(evidence_columns) - set(annotation.columns)
    if missing_evidence:
        raise ValueError(
            f"Annotation file missing evidence columns: {sorted(missing_evidence)}"
        )
    
    ann_core = annotation[evidence_columns].copy()
    
    merged = top.merge(
        ann_core,
        on="pubchem_cid",
        how="left",
        validate="one_to_one",
    )

    merged["novel_rank"] = merged["novel_broad_spectrum_rank"].astype(int)

    missing_annotation = merged[merged["grade"].isna()].copy()

    if len(missing_annotation):
        print("ERROR: Current Top-N contains compounds without annotation:")
        print(
            missing_annotation[
                ["novel_rank", "pubchem_cid", "pubchem_title"]
            ].to_string(index=False)
        )
        raise RuntimeError(
            f"{len(missing_annotation)} current Top-{args.top_n} compounds "
            "require new manual evidence annotation."
        )

    preferred = [
        "novel_rank",
        "pubchem_cid",
        "pubchem_title",
        "SMILES_std",
        "inchikey",
        "mean_predicted_pIC50",
        "median_predicted_pIC50",
        "min_predicted_pIC50",
        "max_predicted_pIC50",
        "n_receptors_above_threshold",
        "grade",
        "evidence_summary",
        "key_source",
        "annotation_note",
    ]

    ordered = preferred + [c for c in merged.columns if c not in preferred]
    merged = merged[ordered]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    out_csv = outdir / "ifra_top50_external_evidence_annotation_latest.csv"
    merged.to_csv(out_csv, index=False)

    counts = (
        merged["grade"]
        .value_counts()
        .reindex(["A", "B", "C", "D", "E"], fill_value=0)
    )

    summary = pd.DataFrame({
        "grade": counts.index,
        "n_compounds": counts.values,
    })
    summary.to_csv(
        outdir / "evidence_grade_summary.csv",
        index=False,
    )

    print("=== Current Top-50 external-evidence annotation ===")
    print(f"Top compounds: {len(merged)}")
    print("\nGrade counts:")
    print(counts.to_string())
    print(f"\nOutput: {out_csv}")


if __name__ == "__main__":
    main()
