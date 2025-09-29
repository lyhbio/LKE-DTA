import json
import argparse
import requests
import pandas as pd
from tqdm import tqdm
import time

UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_BASE = "https://rest.uniprot.org/uniprotkb"
HEADERS = {"User-Agent": "chembl-uniprot-mapper/1.0"}


def chembl_to_uniprot(chembl_id: str):

    url = f"{UNIPROT_SEARCH}?query={chembl_id}&fields=accession&format=json"
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        data = r.json()
        results = data.get("results", [])
        if results:
            return results[0].get("primaryAccession")
    except Exception as e:
        print(f"[ERROR] UniProt search failed {chembl_id}: {e}")
    return None


def get_uniprot_seq(accession: str):

    url = f"{UNIPROT_BASE}/{accession}.fasta"
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.ok:
            lines = r.text.splitlines()
            return "".join(l for l in lines if not l.startswith(">"))
    except Exception as e:
        print(f"[ERROR] UniProt fetch failed {accession}: {e}")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", help="chembl_dta_test.csv")
    parser.add_argument("seq_json", help="seq.json")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    df = df.dropna(subset=["target_chembl_id"])
    unique_targets = df["target_chembl_id"].drop_duplicates()
    s=0
    result = {}
    for tid in tqdm(unique_targets, desc="Fetching target sequences"):
        acc = chembl_to_uniprot(tid)
        seq = get_uniprot_seq(acc) if acc else None
        s+=1
        if s%5==0:
            time.sleep(1.5)
        if seq:
            print(f"[SEQ] {tid} | {acc} | {seq[:50]}... (len={len(seq)})")
        else:
            print(f"[WARN] {tid}  (accession={acc})")

        result[tid] = {"accession": acc, "sequence": seq}
        time.sleep(0.2)

    with open(args.seq_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[finish] {args.seq_json}")


if __name__ == "__main__":
    main()