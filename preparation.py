import pickle
from collections import OrderedDict,defaultdict
import pandas as pd
from utils import *


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

datasets = ['davis','kiba']
for dataset in datasets:
    pro_data = json.load(open('data/'+dataset+'/proteins.txt'),object_pairs_hook=OrderedDict)

    compound_iso_smiles = []
    opts = ['train','test']

    data_sets = ['davis','kiba']
    for data_set in data_sets:
        for opt in opts:
            df = pd.read_csv('data/'+data_set + '_' + opt + '.csv')
            compound_iso_smiles += list(df['compound_iso_smiles'])

    compound_iso_smiles = set(compound_iso_smiles)
    print(len(compound_iso_smiles))


    value=defaultdict()
    pro_embedding=pickle.load(open('data/'+dataset+'/protein_representations.pkl','rb'))
    print(len(pro_embedding))
    i=0
    for _,seq in pro_data.items():
        value[seq]=pro_embedding[i]
        i+=1

    df = pd.read_csv('data/'+dataset+'_train.csv')
    train_drugs,train_prots,train_Y = list(df['compound_iso_smiles']),list(df['target_sequence']),list(df['affinity'])
    for i in range(len(train_prots)):
        train_prots[i]=value[train_prots[i]]
    train_prots = torch.stack(train_prots)

    df = pd.read_csv('data/'+dataset+'_test.csv')
    test_drugs,test_prots,test_Y = list(df['compound_iso_smiles']),list(df['target_sequence']),list(df['affinity'])
    for i in range(len(test_prots)):
        test_prots[i]=value[test_prots[i]]
    test_prots = torch.stack(test_prots)

    train_drugs,train_Y = np.asarray(train_drugs),np.asarray(train_Y)
    test_drugs,test_Y = np.asarray(test_drugs),np.asarray(test_Y)

    print('preparing ',dataset + '_train.pt in pytorch format!')
    train_data = TestbedDataset(root='.',dataset=dataset + '_train',xd=train_drugs,xt=train_prots,y=train_Y
                                        )
    print('preparing ',dataset + '_test.pt in pytorch format!')
    test_data = TestbedDataset(root='.',dataset=dataset + '_test',xd=test_drugs,xt=test_prots,y=test_Y
                                   )





