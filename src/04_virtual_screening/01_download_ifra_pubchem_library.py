#!/usr/bin/env python3
"""
PubChem IFRA downloader v3

Uses PUG-View annotation endpoints (NOT Substance depositor endpoints):
  1) get PubChem annotation headings
  2) identify fragrance-related headings
  3) query those headings with IFRA as the annotation source
  4) paginate all returned annotations
  5) extract linked PubChem CIDs
  6) retrieve compound properties with PUG-REST

This script intentionally saves diagnostic JSON so PubChem schema/source-name
changes can be inspected without guessing.
"""
import argparse, json, re, time
from pathlib import Path
import pandas as pd
import requests

PUGREST = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PUGVIEW = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view"

SOURCE_VARIANTS = [
    "International Fragrance Association (IFRA)",
    "International Fragrance Association",
    "IFRA",
]

def get_json(s, url, params=None, retries=6, timeout=120):
    last = None
    for i in range(retries):
        try:
            r = s.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r.json(), r.url
            last = RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
            if r.status_code in (400,404):
                break
        except Exception as e:
            last = e
        wait = min(2**i, 30)
        print(f"  retry {i+1}/{retries} after {wait}s")
        time.sleep(wait)
    raise RuntimeError(f"Request failed: {url}\n{last}")

def all_strings(x):
    out=[]
    if isinstance(x,dict):
        for k,v in x.items():
            if isinstance(v,(str,int,float)):
                out.append((str(k),str(v)))
            else:
                out.extend(all_strings(v))
    elif isinstance(x,list):
        for v in x: out.extend(all_strings(v))
    return out

def discover_headings(s, out):
    url=f"{PUGREST}/annotations/headings/JSON"
    j, actual=get_json(s,url)
    (out/"pubchem_annotation_headings.json").write_text(
        json.dumps(j,ensure_ascii=False,indent=2),encoding="utf-8")
    vals=[]
    for k,v in all_strings(j):
        if isinstance(v,str) and ("fragrance" in v.lower() or "odor" in v.lower()):
            vals.append(v.strip())
    # Prefer likely IFRA headings first, but retain all discovered fragrance/odor headings.
    seeds=["Fragrance Ingredient","Fragrance Ingredients"]
    headings=list(dict.fromkeys(seeds+vals))
    print("Candidate annotation headings:")
    for h in headings: print(" ",repr(h))
    return headings

def extract_cids(x):
    cids=set()
    if isinstance(x,dict):
        for k,v in x.items():
            kl=str(k).lower()
            # PUG-View commonly places linked IDs under LinkedRecords -> CID.
            if kl in {"cid","cids"}:
                vals=v if isinstance(v,list) else [v]
                for z in vals:
                    if isinstance(z,int) or (isinstance(z,str) and z.isdigit()):
                        cids.add(int(z))
            cids |= extract_cids(v)
    elif isinstance(x,list):
        for v in x: cids |= extract_cids(v)
    return cids

def page_info(j):
    # Accommodate Page/TotalPages at different nesting levels.
    page=None; total=None
    def walk(x):
        nonlocal page,total
        if isinstance(x,dict):
            for k,v in x.items():
                kl=str(k).lower()
                if kl in {"page","pagenumber"} and isinstance(v,(int,float,str)):
                    try: page=int(v)
                    except: pass
                if kl in {"totalpages","totalpage"} and isinstance(v,(int,float,str)):
                    try: total=int(v)
                    except: pass
                walk(v)
        elif isinstance(x,list):
            for v in x: walk(v)
    walk(j)
    return page,total

def query_annotation_combo(s, heading, source, pause):
    url=f"{PUGVIEW}/annotations/heading/JSON"
    params={"heading":heading,"source":source,"page":1}
    try:
        first, actual=get_json(s,url,params=params,retries=2)
    except Exception as e:
        return None, str(e)

    cids=extract_cids(first)
    _, total=page_info(first)
    if total is None: total=1
    pages=[first]
    print(f"    page 1/{total}: {len(cids)} linked CIDs so far")

    for p in range(2,total+1):
        j,_=get_json(s,url,params={"heading":heading,"source":source,"page":p})
        pages.append(j)
        cids |= extract_cids(j)
        print(f"    page {p}/{total}: {len(cids)} linked CIDs so far")
        time.sleep(pause)
    return {"heading":heading,"source":source,"url":actual,
            "total_pages":total,"cids":sorted(cids),"pages":pages}, None

def find_ifra_annotations(s, headings, out, pause):
    attempts=[]
    best=None
    for heading in headings:
        for source in SOURCE_VARIANTS:
            print(f"\nTesting heading={heading!r} source={source!r}")
            result,err=query_annotation_combo(s,heading,source,pause)
            if result is None:
                print("  failed:",err.splitlines()[-1][:300])
                attempts.append({"heading":heading,"source":source,"error":err})
                continue
            n=len(result["cids"])
            print(f"  SUCCESS: {n} unique linked CIDs; pages={result['total_pages']}")
            attempts.append({"heading":heading,"source":source,
                             "n_cids":n,"total_pages":result["total_pages"],
                             "url":result["url"]})
            if n and (best is None or n>len(best["cids"])):
                best=result
            # A valid large IFRA result is enough; avoid unnecessary server load.
            if n >= 500:
                break
        if best is not None and len(best["cids"]) >= 500:
            break

    (out/"ifra_annotation_attempts.json").write_text(
        json.dumps(attempts,ensure_ascii=False,indent=2),encoding="utf-8")
    if best is None:
        raise RuntimeError(
            "No IFRA annotation query returned linked CIDs. "
            "Inspect pubchem_annotation_headings.json and ifra_annotation_attempts.json."
        )
    # Save only the winning raw pages.
    (out/"ifra_annotation_pages.json").write_text(
        json.dumps(best["pages"],ensure_ascii=False),encoding="utf-8")
    return best

def fetch_properties(s,cids,batch,pause):
    props="Title,ConnectivitySMILES,SMILES,InChIKey,MolecularFormula,MolecularWeight"
    rows=[]
    for i in range(0,len(cids),batch):
        b=cids[i:i+batch]
        url=f"{PUGREST}/compound/cid/{','.join(map(str,b))}/property/{props}/JSON"
        j,_=get_json(s,url)
        rows += j.get("PropertyTable",{}).get("Properties",[])
        print(f"Properties: {min(i+batch,len(cids))}/{len(cids)}")
        time.sleep(pause)
    d=pd.DataFrame(rows)
    return d.rename(columns={
        "CID":"pubchem_cid","Title":"pubchem_title",
        "ConnectivitySMILES":"connectivity_smiles","SMILES":"smiles",
        "InChIKey":"inchikey","MolecularFormula":"molecular_formula",
        "MolecularWeight":"molecular_weight"})

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--outdir",default="data/ifra_pubchem_v3")
    ap.add_argument("--batch_size",type=int,default=100)
    ap.add_argument("--pause",type=float,default=0.25)
    ap.add_argument("--heading",default=None,
                    help="Optional exact PUG-View heading; skips heading discovery.")
    ap.add_argument("--source",default=None,
                    help="Optional exact annotation source name.")
    a=ap.parse_args()
    out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True)
    s=requests.Session()
    s.headers["User-Agent"]="OdorBlock-IFRA-PubChem/3.0 academic-research"

    print("=== PubChem IFRA library download v3 ===")
    print("PUG-View annotations -> linked CID -> PUG-REST properties\n")

    global SOURCE_VARIANTS
    if a.source: SOURCE_VARIANTS=[a.source]
    headings=[a.heading] if a.heading else discover_headings(s,out)
    best=find_ifra_annotations(s,headings,out,a.pause)
    cids=best["cids"]

    print("\n=== Winning IFRA annotation query ===")
    print("Heading:",best["heading"])
    print("Source:",best["source"])
    print("Pages:",best["total_pages"])
    print("Unique linked CIDs:",len(cids))

    pd.Series(cids).to_csv(out/"ifra_pubchem_cids.txt",index=False,header=False)
    d=fetch_properties(s,cids,a.batch_size,a.pause)
    if "pubchem_cid" in d.columns:
        d=d.drop_duplicates("pubchem_cid").sort_values("pubchem_cid").reset_index(drop=True)
    d.to_csv(out/"ifra_pubchem_raw.csv",index=False)

    meta={
        "route":"PUG-View annotations by heading/source -> linked CID -> PUG-REST properties",
        "pubchem_source_page_reference":26581,
        "heading":best["heading"],"source":best["source"],
        "total_pages":best["total_pages"],"n_unique_linked_cids":len(cids),
        "n_property_rows":len(d),"rdkit_standardization_applied":False
    }
    (out/"download_metadata.json").write_text(
        json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"\n=== Complete ===\nCIDs: {len(cids)}\nRows: {len(d)}")
    print("CSV:",(out/"ifra_pubchem_raw.csv").resolve())

if __name__=="__main__":
    main()
