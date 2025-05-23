import json
import pandas as pd


with open('data/ligands with chembl id.json', 'r') as f:
    ligands_id=json.load(f)
ligands={}
for i in ligands_id:
    ligands[ligands_id[i]]=i

opts=['train.csv','test.csv']
datasets=['davis','kiba']
for dataset in datasets:
    with open("data/"+dataset+'/'+"training_triplets.tsv",'w') as fi:
        for opt in opts:
            f='data/'+dataset+'_'+opt
            df=pd.read_csv(f)
            if dataset=='davis':
                for i in range(len(df)):
                    smiles=df['compound_iso_smiles'][i]
                    target_name=df['target_name'][i]
                    fi.writelines("{}\t{}\t{}\n".format(ligands[df['compound_iso_smiles'][i]],'/','/'))
            else:
                for i in range(len(df)):
                    if df['drug_name'][i]=='CHEMBL1':
                	    print(i)
                	    print(dataset,opt)
                    fi.writelines(
                        "{}\t{}\t{}\n".format(df['drug_name'][i],'/','/'))
