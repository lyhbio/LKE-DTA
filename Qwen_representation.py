import json
import os
from openai import OpenAI
import dashscope
from http import HTTPStatus

def get_embedding(input_text):
    c = dashscope.TextEmbedding.call(
        model=dashscope.TextEmbedding.Models.text_embedding_v2,
        input=input_text
    )
    return c

all_embedding = {}
s=0
with open('data/IUPAC_name.json','r') as f:
    drug_descriptions = json.load(f)
    for i in drug_descriptions:
        result = get_embedding(drug_descriptions[i])
        status = result['status_code']
        embedding = result['output']['embeddings'][0]['embedding']
        if status != HTTPStatus.OK:
            print('ques')
        all_embedding[i]=embedding
        s+=1
        print(s)

with open('data/Qwen_representation.json', 'w') as f:
    json.dump(all_embedding, f, indent=4)
