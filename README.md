# LKE-DTA
# LKE-DTA: Predicting Drug–Target Binding Affinity

**LKE-DTA** is an innovative framework designed to predict **drug–target binding affinity** by integrating **Large Language Model (LLM) representations** and **Knowledge Graph Embeddings**. By combining natural language processing techniques and graph learning methods, LKE-DTA offers a robust approach for modeling and predicting drug–target interactions with high accuracy.

This program involves two separate environments to extract drug and protein feature representations and model training.

![LKE-DTA Architecture](images/lke-dta-workflow.png)

---

## Feature Embedding Extraction

---

### Environment 1: Drug Feature Embedding

#### **Requirements**
- Python 3.6  
- numpy 1.21.5  
- pandas 1.3.5  
- torch 1.2.0  
- dgl 0.4  

#### **Step 1: Drug Feature Embedding (TransE + iBKH)**

1. **Generate Triplets**  
Run the following script to prepare the knowledge graph triplets:
```bash
python ibkh-tembedding.py
```

2. **Train TransE Model using iBKH Knowledge Graph**

Run the appropriate command based on the dataset you are using:

**For Davis dataset:**
```bash
DGLBACKEND=pytorch dglke_train --dataset iBKH --data_path ./data/davis --data_files training_triplet.tsv --format raw_udd_hrt --model_name TransE_l2 --batch_size 3000 --neg_sample_size 256 --hidden_dim 400 --gamma 12.0 --lr 0.1 --max_step 50000 --log_interval 100 --batch_size_eval 1000 -adv --regularization_coef 1.00E-09 --num_thread 1 --num_proc 8 --neg_sample_size_eval 1000
```

**For KIBA dataset:**
```bash
DGLBACKEND=pytorch dglke_train --dataset iBKH --data_path ./data/kiba --data_files training_triplet.tsv --format raw_udd_hrt --model_name TransE_l2 --batch_size 3000 --neg_sample_size 256 --hidden_dim 400 --gamma 12.0 --lr 0.1 --max_step 50000 --log_interval 100 --batch_size_eval 1000 -adv --regularization_coef 1.00E-09 --num_thread 1 --num_proc 8 --neg_sample_size_eval 1000
```

---

## Semantic Representation Extraction

### Environment 2: Semantic Representation and Model Training

#### **Requirements**
- Python 3.10  
- `esm` 2.0.0  
- `torch` 2.5.1  
- `dashscope`  
- `torch_geometric` 2.6.1  

#### **Step 2: Drug Semantic Representation (Qwen)**

1. **register for the Qwen API and set your API key**
```bash
export DASHSCOPE_API_KEY="YOUR_DASHSCOPE_API_KEY"
```

2. **Run Drug Representation Script**
```bash
python qwen_representation.py
```

---

#### **Step 3: Protein Feature Representation (ESM)**

Using the same environment:
```bash
python esm_representation.py
```

---

## DTA Model Training

### **Step 4: Training**
Utilize the same Python Environment 2 and Use **Distributed Data Parallel (DDP)** for multi-GPU training. Run the training script with 4 GPUs (e.g., 4×3090):

```bash
torchrun --nproc-per-node=4 training.py
```
---

## Conclusion

This README outlines the complete setup and execution pipeline for the LKE-DTA project. For further support or advanced configuration, please consult the repositories for the following components:

- [iBKH](https://github.com/wcm-wanglab/iBKH)
- [Qwen](https://bailian.console.aliyun.com/?tab=api#/api/?type=model&url=https%3A%2F%2Fhelp.aliyun.com%2Fdocument_detail%2F2712515.html)
- [ESM](https://github.com/facebookresearch/esm)
