# Granularity-Aware Experts on hierarchical Graphs for Vietnamese Extractive Multi-Document Summarization
___

This repository contains the implementation of **GranEx-Graph**, a hierarchical graph-based model that integrates a novel Mixture-of-Experts mechanism with Graph Attention Networks (GAT) for Vietnamese text summarization. The model employs granularity-aware experts to capture structural information at sentence, section, and document levels, effectively balancing local coherence and global semantics within a unified graph attention framework.  
___

## Usage

### Data Preprocessing
To preprocess the data, run:
```bash
jupyter nbconvert --to notebook --execute notebook-process/processing_data.ipynb
```
### Training Process and Evaluation Process

```bash
jupyter nbconvert --to notebook --execute notebook-process/training_and_evaluation.ipynb
```
### Inference 
```bash
jupyter nbconvert --to notebook --execute notebook-process/inference.ipynb
```
