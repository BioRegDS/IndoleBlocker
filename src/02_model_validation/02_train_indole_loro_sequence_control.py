#!/usr/bin/env python3
"""
Indole-only LORO-CV: receptor representation control.

Compare:
1) antagonist ECFP4 only
2) antagonist ECFP4 + receptor one-hot
3) antagonist ECFP4 + simple amino-acid composition (AAC; 20 dims)
4) antagonist ECFP4 + ESM-2 embedding

AAC is computed as the fraction of each standard amino acid in the receptor
protein sequence. No learned protein representation is used for AAC.

LORO = leave-one-receptor-out cross-validation.
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


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
    """20-dimensional normalized amino-acid composition."""
    counts = np.array([seq.count(aa) for aa in AA_ORDER], dtype=np.float32)
    denom = counts.sum()
    if denom <= 0:
        raise ValueError("Sequence contains no standard amino acids.")
    return counts / denom


def pearson_r(y, p):
    if len(y) < 2 or np.std(y) == 0 or np.std(p) == 0:
        return np.nan
    return float(np.corrcoef(y, p)[0, 1])


def calc_metrics(y, p):
    return {
        "MAE": float(mean_absolute_error(y, p)),
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "R2": float(r2_score(y, p)),
        "Pearson_r": pearson_r(y, p),
    }


def build_model(kind, seed, n_estimators):
    if kind == "rf":
        return RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=seed,
            n_jobs=-1,
            max_features="sqrt",
        )
    if kind == "svr":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(kernel="rbf", C=1.0, epsilon=0.1, gamma="scale")),
        ])
    if kind == "ridge":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ])
    if kind == "gbr":
        return GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=0.05,
            max_depth=2,
            random_state=seed,
            loss="squared_error",
        )
    raise ValueError(kind)


def load_esm2_embeddings(sequences, model_name, cache_dir, device):
    import torch
    from transformers import AutoTokenizer, AutoModel

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    embeddings = {}
    missing = []

    for seq in list(dict.fromkeys(sequences)):
        key = hashlib.sha256((model_name + "|" + seq).encode()).hexdigest()
        path = cache_dir / f"{key}.npy"
        if path.exists():
            embeddings[seq] = np.load(path)
        else:
            missing.append((seq, path))

    if missing:
        print(f"Loading ESM-2: {model_name}")
        print(f"Device: {device}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        model.eval()

        for i, (seq, path) in enumerate(missing, 1):
            tokens = tokenizer(seq, return_tensors="pt").to(device)
            with torch.no_grad():
                hidden = model(**tokens).last_hidden_state[0, 1:-1]
                emb = hidden.mean(0).cpu().numpy().astype(np.float32)
            embeddings[seq] = emb
            np.save(path, emb)
            print(f"Embedded {i}/{len(missing)}")

    return embeddings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--sheet", default="regression_strict_exact_113")
    ap.add_argument("--outdir", default="results_indole_sequence_control")
    ap.add_argument("--sequence_col", default="receptor_sequence")
    ap.add_argument("--esm_model", default="facebook/esm2_t6_8M_UR50D")
    ap.add_argument("--embedding_cache", default=".cache/odorblock_esm2")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--model", choices=["rf", "svr", "ridge", "gbr", "all"], default="all")
    ap.add_argument("--n_estimators", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(str(args.input))
#    df = pd.read_excel(args.input, sheet_name=args.sheet)

    if "endpoint_type" in df.columns:
        df = df[df["endpoint_type"].astype(str).str.upper() == "IC50"].copy()

    df = df[df["malodor"].astype(str).str.strip().str.lower() == "indole"].copy()

    activity_col = next(
        c for c in ["activity_uM", "IC50_uM", "IC50", "activity_value"]
        if c in df.columns
    )

    df["IC50_uM_numeric"] = pd.to_numeric(df[activity_col], errors="coerce")
    df["blocker_SMILES_std"] = df["blocker_SMILES"].apply(standardize_smiles)
    df["seq"] = df[args.sequence_col].apply(clean_sequence)

    df = df.dropna(
        subset=["receptor", "blocker_SMILES_std", "seq", "IC50_uM_numeric"]
    )
    df = df[df["IC50_uM_numeric"] > 0].reset_index(drop=True)
    df["receptor"] = df["receptor"].astype(str).str.strip()
    df["pIC50"] = 6.0 - np.log10(df["IC50_uM_numeric"])

    # ECFP4
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    X_blocker = np.stack([
        np.asarray(
            fpgen.GetFingerprint(Chem.MolFromSmiles(s)),
            dtype=np.float32
        )
        for s in df["blocker_SMILES_std"]
    ])

    # Unique receptor sequences
    rs = df[["receptor", "seq"]].drop_duplicates()
    bad = rs.groupby("receptor")["seq"].nunique()
    bad = bad[bad > 1]
    if len(bad):
        raise ValueError("Multiple sequences found for: " + ", ".join(bad.index))

    rs = rs.drop_duplicates("receptor").reset_index(drop=True)
    receptors = rs["receptor"].tolist()

    # AAC
    aac_map = {
        r: aa_composition(s)
        for r, s in zip(rs["receptor"], rs["seq"])
    }

    # ESM-2
    seq_embeddings = load_esm2_embeddings(
        rs["seq"].tolist(),
        args.esm_model,
        args.embedding_cache,
        args.device,
    )
    esm_map = {
        r: seq_embeddings[s]
        for r, s in zip(rs["receptor"], rs["seq"])
    }

    y = df["pIC50"].to_numpy()

    estimators = (
        ["rf", "svr", "ridge", "gbr"]
        if args.model == "all"
        else [args.model]
    )

    feature_modes = [
        "antagonist",
        "antagonist_receptor_onehot",
        "antagonist_receptor_aac",
        "antagonist_receptor_esm2",
    ]

    summary_rows = []
    receptor_rows = []

    for estimator in estimators:
        preds = {
            mode: np.full(len(df), np.nan)
            for mode in feature_modes
        }

        for fold_idx, heldout in enumerate(receptors, 1):
            test_idx = np.where(df["receptor"].to_numpy() == heldout)[0]
            train_idx = np.where(df["receptor"].to_numpy() != heldout)[0]

            train_receptors = [r for r in receptors if r != heldout]

            # One-hot must be learned from training receptors only.
            # The unseen held-out receptor therefore becomes an all-zero vector.
            onehot_index = {r: i for i, r in enumerate(train_receptors)}
            X_onehot = np.zeros((len(df), len(train_receptors)), dtype=np.float32)
            for row_i, r in enumerate(df["receptor"]):
                if r in onehot_index:
                    X_onehot[row_i, onehot_index[r]] = 1.0

            X_aac = np.stack([aac_map[r] for r in df["receptor"]]).astype(np.float32)
            X_esm = np.stack([esm_map[r] for r in df["receptor"]]).astype(np.float32)

            feature_matrices = {
                "antagonist": X_blocker,
                "antagonist_receptor_onehot": np.c_[X_blocker, X_onehot],
                "antagonist_receptor_aac": np.c_[X_blocker, X_aac],
                "antagonist_receptor_esm2": np.c_[X_blocker, X_esm],
            }

            for mode in feature_modes:
                model = build_model(
                    estimator,
                    args.seed + fold_idx,
                    args.n_estimators
                )
                X = feature_matrices[mode]
                model.fit(X[train_idx], y[train_idx])
                p = model.predict(X[test_idx])
                preds[mode][test_idx] = p

                mm = calc_metrics(y[test_idx], p)
                receptor_rows.append({
                    "estimator": estimator,
                    "feature_mode": mode,
                    "heldout_receptor": heldout,
                    "n_test": len(test_idx),
                    **mm,
                })

        for mode in feature_modes:
            mm = calc_metrics(y, preds[mode])

            dims = {
                "antagonist": 2048,
                "antagonist_receptor_onehot": 2048 + (len(receptors) - 1),
                "antagonist_receptor_aac": 2048 + 20,
                "antagonist_receptor_esm2": 2048 + len(next(iter(esm_map.values()))),
            }

            summary_rows.append({
                "estimator": estimator,
                "feature_mode": mode,
                "n_records": len(df),
                "unique_blockers": df["blocker_SMILES_std"].nunique(),
                "unique_receptors": df["receptor"].nunique(),
                "n_features_loro_train": dims[mode],
                **mm,
            })

    summary = pd.DataFrame(summary_rows)
    by_receptor = pd.DataFrame(receptor_rows)

    summary.to_csv(outdir / "model_comparison.csv", index=False)
    by_receptor.to_csv(outdir / "per_receptor_model_comparison_long.csv", index=False)

    # Pairwise receptor-wise RMSE improvements versus AAC and antagonist
    wide = by_receptor.pivot_table(
        index=["estimator", "heldout_receptor", "n_test"],
        columns="feature_mode",
        values="RMSE"
    ).reset_index()

    wide["delta_RMSE_aac_vs_antagonist"] = (
        wide["antagonist_receptor_aac"] - wide["antagonist"]
    )
    wide["delta_RMSE_esm2_vs_antagonist"] = (
        wide["antagonist_receptor_esm2"] - wide["antagonist"]
    )
    wide["delta_RMSE_esm2_vs_aac"] = (
        wide["antagonist_receptor_esm2"] - wide["antagonist_receptor_aac"]
    )
    wide["delta_RMSE_esm2_vs_onehot"] = (
        wide["antagonist_receptor_esm2"] - wide["antagonist_receptor_onehot"]
    )
    wide.to_csv(outdir / "per_receptor_rmse_comparison.csv", index=False)

    # Wilcoxon tests on receptor-wise RMSE
    try:
        from scipy.stats import wilcoxon
        stat_rows = []
        comparisons = [
            ("AAC_vs_antagonist", "antagonist_receptor_aac", "antagonist"),
            ("ESM2_vs_antagonist", "antagonist_receptor_esm2", "antagonist"),
            ("ESM2_vs_onehot", "antagonist_receptor_esm2", "antagonist_receptor_onehot"),
            ("ESM2_vs_AAC", "antagonist_receptor_esm2", "antagonist_receptor_aac"),
        ]

        for estimator in estimators:
            q = wide[wide["estimator"] == estimator]
            for name, second, first in comparisons:
                delta = q[second].to_numpy() - q[first].to_numpy()
                try:
                    w = wilcoxon(delta, alternative="two-sided")
                    stat = float(w.statistic)
                    pval = float(w.pvalue)
                except ValueError:
                    stat = np.nan
                    pval = np.nan

                stat_rows.append({
                    "estimator": estimator,
                    "comparison": name,
                    "n_receptors": len(delta),
                    "wilcoxon_statistic": stat,
                    "p_value_two_sided": pval,
                    "median_delta_RMSE": float(np.median(delta)),
                    "n_improved": int((delta < 0).sum()),
                })

        stats = pd.DataFrame(stat_rows)
        stats.to_csv(outdir / "wilcoxon_receptor_rmse.csv", index=False)
    except ImportError:
        stats = None

    # Figure: RMSE by representation
    labels = {
        "antagonist": "Antagonist",
        "antagonist_receptor_onehot": "One-hot",
        "antagonist_receptor_aac": "AA composition",
        "antagonist_receptor_esm2": "ESM-2",
    }

    for estimator in estimators:
        q = summary[summary["estimator"] == estimator].copy()
        q["order"] = pd.Categorical(
            q["feature_mode"],
            categories=feature_modes,
            ordered=True
        )
        q = q.sort_values("order")

        fig, ax = plt.subplots(figsize=(8, 6))
        x = np.arange(len(q))
        vals = q["RMSE"].to_numpy()
        ax.plot(x, vals, marker="o")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [labels[m] for m in q["feature_mode"]],
            rotation=25,
            ha="right"
        )
        ax.set_ylabel("LORO RMSE")
        ax.set_title(
            f"Indole-only receptor representation control ({estimator.upper()})"
        )
        for xi, yi in zip(x, vals):
            ax.text(xi, yi, f"{yi:.3f}", ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        for ext in ["png", "pdf"]:
            fig.savefig(
                outdir / f"figure_receptor_representation_rmse__{estimator}.{ext}",
                dpi=300,
                bbox_inches="tight"
            )
        plt.close(fig)

    print("=== Indole-only receptor representation control ===")
    print(f"Usable rows: {len(df)}")
    print(f"Unique blockers: {df['blocker_SMILES_std'].nunique()}")
    print(f"Unique receptors: {df['receptor'].nunique()}")
    print(f"AAC dimension: 20")
    print(f"ESM-2 dimension: {len(next(iter(esm_map.values())))}")
    print()
    print(summary[
        ["estimator", "feature_mode", "n_features_loro_train",
         "MAE", "RMSE", "R2", "Pearson_r"]
    ].to_string(index=False))

    if stats is not None:
        print()
        print("=== Receptor-wise Wilcoxon signed-rank tests ===")
        print(stats.to_string(index=False))

    print()
    print("Delta RMSE < 0 means the first representation in the comparison improved RMSE.")
    print("Outputs written to:", outdir.resolve())


if __name__ == "__main__":
    main()
