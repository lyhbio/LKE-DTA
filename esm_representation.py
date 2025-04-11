import json
import pickle
import esm
from collections import OrderedDict
import torch

# 加载 ESM 模型

model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 定义批次大小
BATCH_SIZE = 8


for dataset in ['davis','kiba']:

    pro_data = json.load(open(f'data/{dataset}/proteins.txt'), object_pairs_hook=OrderedDict)
    sequence_data = [(name, seq) for name, seq in pro_data.items()]
    sequence_representations = []
    for i in range(0, len(sequence_data), BATCH_SIZE):
        batch_data = sequence_data[i:i + BATCH_SIZE]
        batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)


        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
            token_representations = results["representations"][33]
            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)


            for j, tokens_len in enumerate(batch_lens):
                sequence_representations.append(
                    token_representations[j, 1: tokens_len - 1].mean(0).cpu()
                )

    save_path = f'data/{dataset}/protein_representations.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(sequence_representations, f)
    print(f"{dataset} finished。\n")