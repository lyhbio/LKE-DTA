# -*- coding: utf-8 -*-
"""
1) IUPAC_name.json : { "Smiles": "IUPAC" }
2) seq.json        : { "target_chembl_id": "SEQUENCE" }
    python extract_drug.py chembl_dta_test.csv IUPAC_name.json seq.json
"""

import sys
import time
import json
import argparse
import requests
import pandas as pd
from tqdm import tqdm

CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"
PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
HEADERS = {"User-Agent": "chembl-pubchem-enricher/1.0"}


def _get_json(url: str, max_retries: int = 3, backoff: float = 1.5) -> dict:
    for i in range(max_retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(backoff ** i)
            else:
                r.raise_for_status()
        except Exception:
            time.sleep(backoff ** i)
    return {}


def get_iupac_from_chembl(mol_chembl_id: str):

    data = _get_json(f"{CHEMBL_BASE}/molecule/{mol_chembl_id}.json")
    if not data:
        return None, None
    props = data.get("molecule_properties") or {}
    iupac = props.get("iupac_name")
    if iupac:
        return iupac, None

    inchikey = data.get("molecule_structures", {}).get("standard_inchi_key")
    return None, inchikey




def get_iupac_from_pubchem_inchikey(inchikey: str):

    url = f"{PUBCHEM_BASE}/compound/inchikey/{inchikey}/property/IUPACName/JSON"
    data = _get_json(url)
    props = data.get("PropertyTable", {}).get("Properties", [])
    if props and "IUPACName" in props[0]:
        return props[0]["IUPACName"]
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", help="chembl_dta_test.csv")
    parser.add_argument("iupac_json", help="输出 IUPAC_name.json")
    parser.add_argument("seq_json", help="输出 seq.json")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    df = df.dropna(subset=["drug_chembl_id", "target_chembl_id", "Smiles"])

    # ---- (A) IUPAC_name.json ----
    smiles_to_iupac = {}
    unique_mols = df[["drug_chembl_id", "Smiles"]].drop_duplicates()

    for _, row in tqdm(unique_mols.iterrows(), total=len(unique_mols), desc="Fetching IUPAC"):
        mol_id = row["drug_chembl_id"]
        smi = row["Smiles"]
        iupac_name = None
        try:
            iupac, inchikey = get_iupac_from_chembl(mol_id)
            if not iupac and inchikey:
                iupac = get_iupac_from_pubchem_inchikey(inchikey)
            iupac_name = iupac
        except Exception:
            pass
        smiles_to_iupac[smi] = iupac_name
        print(f"[IUPAC] {mol_id} | {smi[:30]}... -> {iupac_name}")
        time.sleep(0.05)

    with open(args.iupac_json, "w", encoding="utf-8") as f:
        json.dump(smiles_to_iupac, f, ensure_ascii=False, indent=2, sort_keys=True)


    print(f"finish {args.iupac_json} & {args.seq_json}")

if __name__ == "__main__":
    main()