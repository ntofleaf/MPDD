## Requirements
- torch 1.13.1
- pytorch_geometric 2.6.1

## Running

### Step 1: Preprocess.
> python -u preprocess.py

### Step 2: Train. 
> python -u train.py --once

### Step 3: Test.
> python -u val.py --once --view

## Dataset

### Download

The dataset is provided in pickle (.pkl) format and can be directly downloaded from the following link:

- **Dataset Download**: [Download Link](https://drive.google.com/file/d/17Trh3tfeEQ0P6-K29lwW8Q1L9UHyV8sP/view?usp=sharing)

### Usage Agreement

**Important**: Due to privacy and ethical considerations, this dataset is intended for **research purposes only**. Before using the dataset, you must:

1. Read and agree to our Data Usage Agreement
2. Submit the signed agreement form to us
3. Commit to protecting data privacy and using the dataset responsibly

**Data Usage Agreement**: [View and Download Agreement](https://github.com/hacilab/MPDD/blob/main/MPDD%20Dataset%20License%20Agreementt.pdf)

After downloading the dataset, please fill out the agreement form and send it to: `fuchangzeng@qhd.neu.edu.cn`

**Note**: The agreement ensures that the dataset is used in compliance with privacy regulations and ethical research standards.

## Citation

If you use this dataset or code in your research, please cite our work:

```bibtex
@article{fu2025personality,
  title={Personality-guided Public-Private Domain Disentangled Hypergraph-Former Network for Multimodal Depression Detection},
  author={Fu, Changzeng and Zhao, Shiwen and Zhang, Yunze and Jian, Zhongquan and Zhao, Shiqi and Liu, Chaoran},
  journal={arXiv preprint arXiv:2511.12460},
  year={2025}
}
```
