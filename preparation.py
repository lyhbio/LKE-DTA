import pickle
from collections import OrderedDict,defaultdict
import pandas as pd
from utils import *
import random

from collections import OrderedDict

import numpy as np
import json
import pandas as pd



def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def split_into_folds(n_samples: int, n_splits: int, seed: int):
    rng = np.random.RandomState(seed)
    idx = np.arange(n_samples)
    rng.shuffle(idx)
    folds = np.array_split(idx, n_splits)
    return folds

def main():
    random_seed = 42
    N_SPLITS = 5
    datasets = ["davis", "kiba"]
    set_seed(random_seed)

    for dataset in datasets:
        train_file = f"data/{dataset}_train.csv"
        test_file = f"data/{dataset}_test.csv"

        if not os.path.exists(train_file) or not os.path.exists(test_file):
            print(f"[WARN] Missing {train_file} or {test_file}, skip {dataset}.")
            continue

        os.makedirs(f"data/{dataset}", exist_ok=True)

        df_train = pd.read_csv(train_file)
        df_test = pd.read_csv(test_file)
        df_full = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)

        n_samples = len(df_full)
        folds = split_into_folds(n_samples, N_SPLITS, random_seed)

        for fold_id, fold_idx in enumerate(folds, start=1):
            df_fold = df_full.loc[fold_idx].reset_index(drop=True)
            out_file = f"data/{dataset}/{dataset}_fold{fold_id}.csv"
            df_fold.to_csv(out_file, index=False)
            print(f"[INFO] Saved {dataset} fold {fold_id} -> {out_file} ({len(df_fold)} samples)")

                  
    embed_davis='ckpts/TransE_l2_iBKH_0/iBKH_TransE_l2_entity.npy'
    embed_kiba='ckpts/TransE_l2_iBKH_1/iBKH_TransE_l2_entity.npy'

    rela_davis='data/davis/entities.tsv'
    rela_kiba='data/kiba/entities.tsv'

    e_d=np.load(embed_davis)
    e_k=np.load(embed_kiba)

    r_d=pd.read_csv(rela_davis,sep='\t',header=None).values
    r_k=pd.read_csv(rela_kiba,sep='\t',header=None).values

    drug_embed={}
    prot_embed={}
    dict_davis=json.load(open('data/ligands with chembl id.json'),object_pairs_hook=OrderedDict)
    dict_kiba=json.load(open('data/kiba/ligands_can.txt'),object_pairs_hook=OrderedDict)


    for i in range(len(e_d)):
        if 'CHEMBL' in r_d[i][1] and r_d[i][1] not in drug_embed:
            drug_embed[dict_davis[r_d[i][1]]]=e_d[i].tolist()
        elif 'CHEMBL' not in r_d[i][1] and r_d[i][1] not in prot_embed:
            prot_embed[r_d[i][1]]=e_d[i].tolist()

    for i in range(len(e_k)):
        if 'CHEMBL' in r_k[i][1] and r_k[i][1] not in drug_embed:
            drug_embed[dict_kiba[r_k[i][1]]]=e_k[i].tolist()
        elif 'CHEMBL' not in r_k[i][1] and r_k[i][1] not in prot_embed:
            prot_embed[r_k[i][1]]=e_k[i].tolist()

    with open('data/ibkh-drug_embedding.json','w') as fi:
        json.dump(drug_embed,fi)

    datasets = ['davis', 'kiba']
    for dataset in datasets:
        pro_data = json.load(open(f'data/{dataset}/proteins.txt'), object_pairs_hook=OrderedDict)

        
        compound_iso_smiles = []
        for fold in range(5):
            df = pd.read_csv(f'data/{dataset}/{dataset}_fold{fold+1}.csv')
            compound_iso_smiles += list(df['compound_iso_smiles'])

        compound_iso_smiles = set(compound_iso_smiles)
        print(dataset, 'unique compounds:', len(compound_iso_smiles))

     
        value = defaultdict()
        pro_embedding = pickle.load(open(f'data/{dataset}/protein_representations.pkl', 'rb'))
        print(dataset, 'protein embeddings:', len(pro_embedding))
        i = 0
        for _, seq in pro_data.items():
            value[seq] = pro_embedding[i]
            i += 1

        for fold in range(5):
            print(f"=== {dataset} fold {fold} ===")
            
            df = pd.read_csv(f'data/{dataset}/{dataset}_fold{fold+1}.csv')
            train_drugs, train_prots, train_Y = list(df['compound_iso_smiles']), list(df['target_sequence']), list(df['affinity'])
            for i in range(len(train_prots)):
                train_prots[i] = value[train_prots[i]]
            train_prots = torch.stack(train_prots)
            train_drugs, train_Y = np.asarray(train_drugs), np.asarray(train_Y)

            print('preparing ', dataset + f'_fold{fold+1}_train.pt in pytorch format!')
            train_data = TestbedDataset(root='.', dataset=dataset + f'_fold{fold}',
                                        xd=train_drugs, xt=train_prots, y=train_Y)

            

if __name__ == "__main__":
    main()
