# G2MoE: Granularity-Aware Graph Mixture-of-Experts for Vietnamese Multi-Document Summarization

This repository contains the implementation of **G2MoE**, a novel framework for Vietnamese multi-document summarization. G2MoE combines heterogeneous graph-based text modeling with a Mixture-of-Experts mechanism, where each expert is implemented using Graph Attention Networks. The model features a Holistic Expert for global context and Granularity-specific Experts for finer-grained semantic levels, enabling effective information synthesis from multiple documents in low-resource settings.

---

## Data

This project uses three publicly available datasets for Vietnamese Multi-Document Summarization:

- **AbmuSu** ([VLSP 2022 Shared Task](https://vlsp.org.vn/vlsp2022/eval/mds)): Contains 200 samples for training and 300 for testing (150 have hidden labels as private test set). For the hidden test set, please contact the organization for data access.
- **ViMs** ([GitHub Repository](https://github.com/CLC-HCMUS/ViMs-Dataset)): We divided samples into 200 training and 100 testing samples
- **VietnameseMDS** ([GitHub Repository](https://github.com/lupanh/VietnameseMDS)): We divided samples into 150 training and 50 testing samples

**Note**: Since summaries are rewritten rather than sentence-selected, sentence labels are based on ROUGE-2 Precision between each sentence and the summary.

### Dataset Format

Place your formatted datasets in the `data/formatted/` directory. Each dataset should be in JSONL format with the following structure:
```json
{
  "single_documents": [
    {
      "raw_text": "Document text...",
      "sentences": ["Sentence 1", "Sentence 2", ...]
    }
  ],
  "summary": "Summary text..."
}
```

---

## Usage

### 1. Data Preprocessing

First, preprocess your data and construct graph structures:
```bash
jupyter nbconvert --to notebook --execute G2MoE/Preprocessing/creating-graph.ipynb
```

**Configuration in the notebook:**
```python
input_path = 'data/formatted/your_dataset.jsonl'
lda_model_path = None  # Set to existing model path to reuse
save_prefix = 'output/graphs'
samples_per_group = 500
n_components = 4  # Number of LDA topics
```

### 2. Training and Evaluation

Train the G2MoE model and evaluate on test set:
```bash
jupyter nbconvert --to notebook --execute G2MoE/model/training_model.ipynb
```

**Key hyperparameters:**
```python
args = {
    'gpu': 0,
    'seed': 42,
    'batch_size': 1,
    'input': 768,
    'hidden': 2048,
    'heads': 128,
    'epochs': 100,
    'lr': 0.0003,
    'dropout': 0.3,
    'num_layers': 3,
    'max_word_num': 185,
    'kappa': 0.2,
    'target_main_contribution': 0.6
}

# if use_dynamic_topk is True --> top_k is the maximum of available topk
moe_args = {
    'num_experts': 3,
    'top_k': 3,
    'use_dynamic_topk': True
}

triple_args = {
    'margin': 0.5,
    'topk': 5,
    'thred': [0.4, 0.5]
}
```

### 3. Output

The model generates:
- Trained model checkpoints (`.mdl` files)
- Summary predictions (`predicts.txt`)
- Reference summaries (`goldens.txt`)
- Evaluation metrics logged to console and Wandb

---

## Project Structure
```
G2MoE/
├── Preprocessing/
│   ├── __init__.py
│   └── creating-graph.ipynb          # Data preprocessing
├── model/
│   ├── __init__.py
│   ├── Graph_Constructor.py          # Graph structure
│   ├── Graph_Encoder.py              # GAT layers
│   ├── MoEGAT.py                     # MoE implementation
│   ├── Loss.py                       # Loss functions
│   ├── Training_Process.py           # Training loop
│   ├── Testing_Process.py            # Inference
│   └── training_model.ipynb          # Training notebook
├── utils.py                          # Utility functions
└── data/
    └── formatted/                    # Place datasets here
```