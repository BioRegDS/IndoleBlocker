#!/usr/bin/env python3
import argparse, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

def std_smiles(s):
    try:
        m=Chem.MolFromSmiles(str(s)); m=rdMolStandardize.FragmentParent(m)
        return Chem.MolToSmiles(m,canonical=True,isomericSmiles=True)
    except: return None

def corr(y,p):
    return np.nan if len(y)<2 or np.std(y)==0 or np.std(p)==0 else np.corrcoef(y,p)[0,1]

def enc():
    try: return OneHotEncoder(handle_unknown="ignore",sparse_output=False)
    except TypeError: return OneHotEncoder(handle_unknown="ignore",sparse=False)

def esm(seqs,name,cache,device):
    import torch
    from transformers import AutoTokenizer,AutoModel
    cache=Path(cache); cache.mkdir(parents=True,exist_ok=True)
    device=("cuda" if torch.cuda.is_available() else "cpu") if device=="auto" else device
    unique=list(dict.fromkeys(seqs)); E={}; missing=[]
    for s in unique:
        f=cache/(hashlib.sha256((name+s).encode()).hexdigest()+".npy")
        if f.exists(): E[s]=np.load(f)
        else: missing.append(s)
    if missing:
        tok=AutoTokenizer.from_pretrained(name); model=AutoModel.from_pretrained(name).to(device); model.eval()
        for s in missing:
            t=tok(s,return_tensors="pt").to(device)
            with torch.no_grad(): h=model(**t).last_hidden_state[0,1:-1].mean(0).cpu().numpy().astype("float32")
            E[s]=h; np.save(cache/(hashlib.sha256((name+s).encode()).hexdigest()+".npy"),h)
    return np.stack([E[s] for s in seqs])

def met(y,p):
    return dict(MAE=mean_absolute_error(y,p),RMSE=np.sqrt(mean_squared_error(y,p)),
                R2=r2_score(y,p) if len(y)>1 else np.nan,Pearson_r=corr(y,p))


def build_regressor(model_type, seed=42, n_estimators=500):
    if model_type == "rf":
        return RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=seed,
            n_jobs=-1,
            max_features="sqrt"
        )

    if model_type == "svr":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(
                kernel="rbf",
                C=1.0,
                epsilon=0.1,
                gamma="scale"
            ))
        ])

    if model_type == "ridge":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0))
        ])

    if model_type == "gbr":
        return GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=0.05,
            max_depth=2,
            random_state=seed,
            loss="squared_error"
        )

    raise ValueError(f"Unknown model type: {model_type}")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True); ap.add_argument("--sheet",default="regression_strict_exact_113")
    ap.add_argument("--outdir",default="results_indole_loro"); ap.add_argument("--sequence_col",default="receptor_sequence")
    ap.add_argument("--esm_model",default="facebook/esm2_t6_8M_UR50D"); ap.add_argument("--embedding_cache",default=".cache/odorblock_esm2")
    ap.add_argument("--device",default="auto"); ap.add_argument("--n_estimators",type=int,default=500); ap.add_argument("--seed",type=int,default=42)
    ap.add_argument(
        "--model",
        choices=["rf","svr","ridge","gbr","all"],
        default="rf",
        help="Regression algorithm: rf, svr, ridge, gbr, or all."
    )
    a=ap.parse_args(); out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True)
    d = pd.read_csv(str(a.input))
    if "endpoint_type" in d: d=d[d.endpoint_type.astype(str).str.upper()=="IC50"].copy()
    d=d[d.malodor.astype(str).str.strip().str.lower()=="indole"].copy()
    activity=next(c for c in ["activity_uM","IC50_uM","IC50","activity_value"] if c in d)
    d["IC50_uM_numeric"]=pd.to_numeric(d[activity],errors="coerce")
    d["blocker_SMILES_std"]=d.blocker_SMILES.apply(std_smiles)
    d["receptor_sequence_clean"]=d[a.sequence_col].astype(str).str.replace(r"\s+","",regex=True)
    d=d.dropna(subset=["blocker_SMILES_std","receptor","receptor_sequence_clean","IC50_uM_numeric"])
    d=d[d.IC50_uM_numeric>0].reset_index(drop=True); d["pIC50"]=6-np.log10(d.IC50_uM_numeric)
    fg=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
    Xb=np.stack([np.asarray(fg.GetFingerprint(Chem.MolFromSmiles(s)),dtype="float32") for s in d.blocker_SMILES_std])
    Xe=esm(d.receptor_sequence_clean.tolist(),a.esm_model,a.embedding_cache,a.device)
    y=d.pIC50.to_numpy(); receptors=d.receptor.drop_duplicates().tolist()
    feature_names=["antagonist","antagonist_receptor_onehot","antagonist_receptor_esm2"]
    estimator_types=["rf","svr","ridge","gbr"] if a.model=="all" else [a.model]

    all_summary=[]
    per_estimator_tables={}

    for estimator_type in estimator_types:
        oof={n:np.full(len(d),np.nan) for n in feature_names}
        rows={n:[] for n in feature_names}

        for k,r in enumerate(receptors,1):
            te=np.where(d.receptor.to_numpy()==r)[0]
            tr=np.where(d.receptor.to_numpy()!=r)[0]

            e=enc()
            Rtr=e.fit_transform(d.loc[tr,["receptor"]])
            Rte=e.transform(d.loc[te,["receptor"]])

            sets={
                "antagonist":(Xb[tr],Xb[te]),
                "antagonist_receptor_onehot":(
                    np.c_[Xb[tr],Rtr],
                    np.c_[Xb[te],Rte]
                ),
                "antagonist_receptor_esm2":(
                    np.c_[Xb[tr],Xe[tr]],
                    np.c_[Xb[te],Xe[te]]
                )
            }

            for n,(A,B) in sets.items():
                m=build_regressor(
                    estimator_type,
                    seed=a.seed+k,
                    n_estimators=a.n_estimators
                )
                m.fit(A,y[tr])
                p=m.predict(B)
                oof[n][te]=p
                rows[n].append(
                    dict(
                        heldout_receptor=r,
                        n_train=len(tr),
                        n_test=len(te),
                        **met(y[te],p)
                    )
                )

        # Save outputs per estimator/feature mode
        for n in feature_names:
            run_name = f"{n}__{estimator_type}" if a.model=="all" else n
            md=out/run_name
            md.mkdir(exist_ok=True)

            pd.DataFrame(rows[n]).to_csv(
                md/"loro_metrics_by_receptor.csv",
                index=False
            )

            q=d.copy()
            q["predicted_pIC50"]=oof[n]
            q["residual"]=q.pIC50-oof[n]
            q["estimator"]=estimator_type
            q["feature_mode"]=n
            q.to_csv(
                md/"oof_predictions.csv",
                index=False
            )

            all_summary.append(
                dict(
                    estimator=estimator_type,
                    feature_mode=n,
                    model=run_name,
                    n_records=len(d),
                    unique_blockers=d.blocker_SMILES_std.nunique(),
                    unique_receptors=d.receptor.nunique(),
                    **met(y,oof[n])
                )
            )

        # Side-by-side per-receptor table for this estimator
        merged=None
        for n in feature_names:
            x=pd.DataFrame(rows[n])[
                ["heldout_receptor","n_test","MAE","RMSE","R2","Pearson_r"]
            ]
            x=x.rename(
                columns={
                    c:f"{c}_{n}"
                    for c in ["MAE","RMSE","R2","Pearson_r"]
                }
            )
            merged=x if merged is None else merged.merge(
                x,
                on=["heldout_receptor","n_test"]
            )

        merged["delta_RMSE_esm2_vs_antagonist"]=(
            merged.RMSE_antagonist_receptor_esm2
            - merged.RMSE_antagonist
        )
        merged["delta_RMSE_esm2_vs_onehot"]=(
            merged.RMSE_antagonist_receptor_esm2
            - merged.RMSE_antagonist_receptor_onehot
        )

        per_estimator_tables[estimator_type]=merged
        merged.to_csv(
            out/f"per_receptor_model_comparison__{estimator_type}.csv",
            index=False
        )

    S=pd.DataFrame(all_summary)
    S.to_csv(out/"model_comparison.csv",index=False)

    # Combined robustness summary by estimator
    robustness_rows=[]
    for estimator_type, merged in per_estimator_tables.items():
        robustness_rows.append({
            "estimator": estimator_type,
            "esm2_better_than_antagonist_n": int(
                (merged.delta_RMSE_esm2_vs_antagonist<0).sum()
            ),
            "esm2_better_than_onehot_n": int(
                (merged.delta_RMSE_esm2_vs_onehot<0).sum()
            ),
            "n_receptors": len(merged),
            "median_delta_RMSE_esm2_vs_antagonist": float(
                merged.delta_RMSE_esm2_vs_antagonist.median()
            ),
            "median_delta_RMSE_esm2_vs_onehot": float(
                merged.delta_RMSE_esm2_vs_onehot.median()
            )
        })

    robustness=pd.DataFrame(robustness_rows)
    robustness.to_csv(
        out/"loro_robustness_summary.csv",
        index=False
    )

    print("=== Indole-only LORO-CV ===")
    print(f"Usable rows: {len(d)}")
    print(f"Unique blockers: {d.blocker_SMILES_std.nunique()}")
    print(f"Unique receptors: {d.receptor.nunique()}")
    print(f"ESM-2 dimension: {Xe.shape[1]}")
    print(f"Estimator selection: {a.model}\n")

    print(
        S[
            [
                "estimator",
                "feature_mode",
                "MAE",
                "RMSE",
                "R2",
                "Pearson_r"
            ]
        ].to_string(index=False)
    )

    print("\n=== LORO robustness summary ===")
    print(robustness.to_string(index=False))

    for estimator_type, merged in per_estimator_tables.items():
        print(f"\n--- {estimator_type} ---")
        print(
            f"ESM-2 improved RMSE vs antagonist: "
            f"{(merged.delta_RMSE_esm2_vs_antagonist<0).sum()}/{len(merged)} receptors"
        )
        print(
            f"ESM-2 improved RMSE vs one-hot: "
            f"{(merged.delta_RMSE_esm2_vs_onehot<0).sum()}/{len(merged)} receptors"
        )

    print(f"\nOutputs written to: {out.resolve()}")

if __name__=="__main__": main()
