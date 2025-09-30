# LKE-DTA
# LKE-DTA: Predicting Drug–Target Binding Affinity

**LKE-DTA** is an advanced framework for predicting **drug–target binding affinity** by integrating **Large Language Model (LLM) representations** with **Knowledge Graph Embeddings**. By combining natural language processing techniques and graph learning methods, LKE-DTA provides a robust and accurate approach to modeling and predicting drug–target interactions.

This project uses two separate environments: one for extracting drug and protein feature representations, and another for model training.



### Download Raw Data

Because of the large dataset size, please run Git LFS after cloning:

```bash
git lfs pull
```


### Step 1: Randomly Split and Extract Drug and Target Data

Run the following script to split the dataset randomly:
```bash
python Generate_test_data.py
```

Then run the scripts below to extract the drug’s IUPAC name and the target’s sequence:
```bash
python extract_drug.py
python extract_seq.py
```


### Step 2: Testing

After preparing the data, you can extract features and start testing by following the instructions in the main README.
