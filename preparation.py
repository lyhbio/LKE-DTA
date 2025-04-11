import pickle
from collections import OrderedDict,defaultdict
import pandas as pd
from utils import *

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





