import json
import pickle
import esm
from collections import OrderedDict
import torch

# ========= 读取数据集 =========
with open("target_sequence.json", "r", encoding="utf-8") as f:
    seq_map = json.load(f)

sequence_data = []
for tid, val in seq_map.items():
    if isinstance(val, dict):
        seq = val.get("sequence")
    else:
        seq = val
    if seq:
        sequence_data.append((tid, seq))

print(f"[INFO] 共载入 {len(sequence_data)} 条序列")

# ========= 加载 ESM 模型 =========
model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()

# 强制使用 cuda:3
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ========= 定义批次大小 =========
BATCH_SIZE = 1
sequence_representations = []

# ========= 前向计算 =========
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

# ========= 保存结果 =========
save_path = "test_protein_representations.pkl"
with open(save_path, "wb") as f:
    pickle.dump(sequence_representations, f)

print(f"[完成] 已提取 {len(sequence_representations)} 条序列的表示，保存到 {save_path}")