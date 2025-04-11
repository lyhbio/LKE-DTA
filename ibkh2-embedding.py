from collections import OrderedDict

import numpy as np
import json
import pandas as pd

embed_davis='data/davis/TransE_l2_iBKH/iBKH_TransE_l2_entity.npy'
embed_kiba='data/kiba/TransE_l2_iBKH/iBKH_TransE_l2_entity.npy'

rela_davis='data/davis/TransE_l2_iBKH/entities.tsv'
rela_kiba='data/kiba/TransE_l2_iBKH/entities.tsv'

e_d=np.load(embed_davis)
e_k=np.load(embed_kiba)

r_d=pd.read_csv(rela_davis,sep='\t',header=None).values
r_k=pd.read_csv(rela_kiba,sep='\t',header=None).values

drug_embed={}
prot_embed={}
dict_davis=json.load(open('ligands with chembl id.json'),object_pairs_hook=OrderedDict)
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

with open('ibkh-drug_embedding.json','w') as fi:
    json.dump(drug_embed,fi)








