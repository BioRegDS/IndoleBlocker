#!/usr/bin/env python3
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

AA=list("ACDEFGHIKLMNPQRSTVWY")

def std_smi(s):
    try:
        m=Chem.MolFromSmiles(str(s))
        if m is None:return None
        m=rdMolStandardize.FragmentParent(m)
        m=rdMolStandardize.Normalizer().normalize(m)
        return Chem.MolToSmiles(m,canonical=True,isomericSmiles=True)
    except:return None

def clean_seq(x):
    if pd.isna(x):
        return None

    s = "".join(str(x).split()).upper()

    if not s:
        return None

    valid_aa = set(AA) | set("XBZUO")
    valid_fraction = sum(c in valid_aa for c in s) / len(s)

    return s if valid_fraction >= 0.95 else None


def aac(s):
    z=np.array([s.count(a) for a in AA],dtype=np.float32)
    return z/z.sum()

def model(seed,n):
    return GradientBoostingRegressor(n_estimators=n,learning_rate=.05,max_depth=2,
                                     random_state=seed,loss="squared_error")

def rmse(y,p):return float(np.sqrt(mean_squared_error(y,p)))

def loro(df,Xb,y,recs,amap,seed,n,drop=None,native=False):
    Xa=np.stack([amap[r] for r in df.receptor])
    keep=[i for i,a in enumerate(AA) if a!=drop]
    X=np.c_[Xb,Xa[:,keep]]
    pred=np.full(len(df),np.nan); rows=[]
    rr=df.receptor.to_numpy()
    for k,h in enumerate(recs,1):
        te=np.where(rr==h)[0]; tr=np.where(rr!=h)[0]
        m=model(seed+k,n);m.fit(X[tr],y[tr]);pred[te]=m.predict(X[te])
        if native:
            for a,v in zip([AA[i] for i in keep],m.feature_importances_[2048:]):
                rows.append([h,a,float(v)])
    return pred,pd.DataFrame(rows,columns=["heldout_receptor","amino_acid","native_importance"])

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True);p.add_argument("--sheet",default="regression_strict_exact_113")
    p.add_argument("--outdir",default="results_indole_aac_interpretability")
    p.add_argument("--sequence_col",default="receptor_sequence")
    p.add_argument("--n_estimators",type=int,default=500);p.add_argument("--n_repeats",type=int,default=200)
    p.add_argument("--seed",type=int,default=42);a=p.parse_args()
    out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(str(a.input))
    if "endpoint_type" in d:d=d[d.endpoint_type.astype(str).str.upper()=="IC50"].copy()
    d=d[d.malodor.astype(str).str.strip().str.lower()=="indole"].copy()
    ac=next(c for c in ["activity_uM","IC50_uM","IC50","activity_value"] if c in d)
    d["v"]=pd.to_numeric(d[ac],errors="coerce");d["smi"]=d.blocker_SMILES.apply(std_smi)
    d["seq"]=d[a.sequence_col].apply(clean_seq)
    d=d.dropna(subset=["receptor","smi","seq","v"]);d=d[d.v>0].reset_index(drop=True)
    d["receptor"]=d.receptor.astype(str).str.strip();d["pIC50"]=6-np.log10(d.v)
    fg=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
    Xb=np.stack([np.asarray(fg.GetFingerprint(Chem.MolFromSmiles(s)),dtype=np.float32) for s in d.smi])
    rs=d[["receptor","seq"]].drop_duplicates()
    bad=rs.groupby("receptor").seq.nunique();bad=bad[bad>1]
    if len(bad):raise ValueError("Multiple sequences: "+",".join(bad.index))
    rs=rs.drop_duplicates("receptor");recs=rs.receptor.tolist()
    amap={r:aac(s) for r,s in zip(rs.receptor,rs.seq)};y=d.pIC50.to_numpy();rr=d.receptor.to_numpy()
    pd.DataFrame([[r]+list(amap[r]) for r in recs],columns=["receptor"]+AA).to_csv(out/"receptor_aac.csv",index=False)

    # Full AAC and native importance
    pf,nat=loro(d,Xb,y,recs,amap,a.seed,a.n_estimators,native=True)
    full=rmse(y,pf)
    nat.to_csv(out/"native_importance_by_fold.csv",index=False)

    # Held-out replacement importance
    rng=np.random.default_rng(a.seed); pr=[]
    Xa=np.stack([amap[r] for r in d.receptor]);X=np.c_[Xb,Xa]
    for k,h in enumerate(recs,1):
        te=np.where(rr==h)[0];tr=np.where(rr!=h)[0];trainr=[r for r in recs if r!=h]
        m=model(a.seed+k,a.n_estimators);m.fit(X[tr],y[tr]);base=mean_squared_error(y[te],m.predict(X[te]))
        for j,aa in enumerate(AA):
            vals=np.array([amap[r][j] for r in trainr]);ds=[]
            for _ in range(a.n_repeats):
                xp=X[te].copy();xp[:,2048+j]=rng.choice(vals)
                ds.append(mean_squared_error(y[te],m.predict(xp))-base)
            pr.append([h,aa,np.mean(ds),np.median(ds),np.mean(np.array(ds)>0)])
    pr=pd.DataFrame(pr,columns=["heldout_receptor","amino_acid","mean_delta_MSE","median_delta_MSE","fraction_worse"])
    pr.to_csv(out/"heldout_replacement_importance.csv",index=False)

    # Leave-one-AAC-feature-out ablation
    ab=[]
    for aa in AA:
        pp,_=loro(d,Xb,y,recs,amap,a.seed,a.n_estimators,drop=aa)
        r=rmse(y,pp);ab.append([aa,r,r-full])
    ab=pd.DataFrame(ab,columns=["amino_acid","RMSE_without_feature","delta_RMSE_vs_full_AAC"])
    ab.to_csv(out/"leave_one_aac_out_ablation.csv",index=False)

    ns=nat.groupby("amino_acid").native_importance.mean().rename("mean_native_importance")
    ps=pr.groupby("amino_acid").mean_delta_MSE.mean().rename("mean_heldout_delta_MSE")
    S=pd.concat([ns,ps],axis=1).reset_index().merge(ab,on="amino_acid")
    S["rank_native"]=S.mean_native_importance.rank(ascending=False)
    S["rank_heldout"]=S.mean_heldout_delta_MSE.rank(ascending=False)
    S["rank_ablation"]=S.delta_RMSE_vs_full_AAC.rank(ascending=False)
    S["mean_rank"]=S[["rank_native","rank_heldout","rank_ablation"]].mean(axis=1)
    S=S.sort_values("mean_rank");S.to_csv(out/"aac_importance_summary.csv",index=False)

    q=ab.sort_values("delta_RMSE_vs_full_AAC")
    fig,ax=plt.subplots(figsize=(8,7));ax.barh(q.amino_acid,q.delta_RMSE_vs_full_AAC);ax.axvline(0)
    ax.set_xlabel("Delta RMSE after removing AAC feature");ax.set_title("AAC feature ablation under LORO-CV")
    fig.tight_layout();fig.savefig(out/"figure_aac_ablation.png",dpi=300);fig.savefig(out/"figure_aac_ablation.pdf");plt.close(fig)

    q=S.sort_values("mean_heldout_delta_MSE")
    fig,ax=plt.subplots(figsize=(8,7));ax.barh(q.amino_acid,q.mean_heldout_delta_MSE);ax.axvline(0)
    ax.set_xlabel("Mean increase in held-out MSE after replacement");ax.set_title("Held-out receptor AAC importance")
    fig.tight_layout();fig.savefig(out/"figure_aac_heldout_importance.png",dpi=300);fig.savefig(out/"figure_aac_heldout_importance.pdf");plt.close(fig)

    print("=== Indole AAC + GBR interpretability ===")
    print(f"Rows: {len(d)} | blockers: {d.smi.nunique()} | receptors: {len(recs)}")
    print(f"Full AAC LORO RMSE: {full:.6f}")
    print("\n=== Importance summary ===")
    print(S[["amino_acid","mean_native_importance","mean_heldout_delta_MSE","delta_RMSE_vs_full_AAC","mean_rank"]].to_string(index=False))
    print("\nPositive heldout delta MSE and positive ablation delta RMSE both support usefulness.")
    print("Caution: 10 receptors and correlated AAC features; treat as exploratory, not residue-level causality.")
    print("Outputs written to:",out.resolve())
if __name__=="__main__":main()
