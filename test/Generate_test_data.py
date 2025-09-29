# -*- coding: utf-8 -*-
"""
直接合并 IUPAC_name.json 和 target_sequence.json 到 chembl_dta_test.csv
新增列：iupac_name, sequence
"""

import json
import pandas as pd

# 输入文件
input_csv = "chembl_dta_test.csv"
iupac_json = "IUPAC_name.json"
seq_json = "target_sequence.json"

# 输出文件
out_csv = "test.csv"

# 读取数据
df = pd.read_csv(input_csv)

with open(iupac_json, "r", encoding="utf-8") as f:
    iupac_map = json.load(f)

with open(seq_json, "r", encoding="utf-8") as f:
    seq_map = json.load(f)

df["iupac_name"] = df["Smiles"].map(iupac_map)

def get_seq(tid):
    val = seq_map.get(tid)
    if isinstance(val, dict):
        return val.get("sequence")
    return val

df["sequence"] = df["target_chembl_id"].map(get_seq)

# 保存结果
df.to_csv(out_csv, index=False)
print(f"[finish]  {out_csv}")


df=pd.read_csv('test.csv')
js=json.load(open('target_sequence.json'))

lst=df['target_chembl_id'].tolist()
app=[]
for i in range(len(lst)):
    app.append(js[lst[i]]['accession'])
df['pdb_code']=app
df.to_csv('drug_test.csv',index=False)