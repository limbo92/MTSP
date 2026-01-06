# MTSP: Multi-Task Scale Prediction for Depression Assessment

This repository contains the implementation of multi-task learning models for depression assessment using PHQ-8 and HAMD-13 scales.

## Directory Structure

```
MTSP/
├── data/                          # Dataset files
│   └── pdch/                      # PDCH dataset (HAMD-13)
│       ├── pdch_original_*.json   # Original format data files
│       └── pdch_summary_*.json    # Summarized format data files
│
├── models/                        # Model implementations
│   ├── multi_scale_transformer.py # Multi-scale Transformer model for depression assessment
│   └── task_graph_gat.py          # Task Graph GAT (Graph Attention Network) for task relationship modeling
│
├── preprocessing/                 # Data loading and preprocessing modules
│   ├── HAMD13Dataset.py           # Dataset loader for HAMD-13 (CIDH, PDCH)
│   └── EDAICDataset.py            # Dataset loader for PHQ-8 (E-DAIC)
│
├── utils/                         # Utility modules
│   ├── multi_scale_config.py      # Unified configuration for multi-scale training
│   ├── cluster_constraint_loss.py # Cluster constraint loss based on clinical symptom clusters
│   ├── task_level_spl.py          # Task-Level Self-Paced Learning (SPL) implementation
│   ├── ordinal_loss.py            # Ordinal classification loss
│   ├── early_stopping.py          # Early stopping callback
│   ├── utils.py                   # General utility functions
│   ├── playground_HAMD13.py       # Playground script for LLM evaluation on HAMD-13
│   └── playground_PHQ8.py         # Playground script for LLM evaluation on PHQ-8
│
├── train_multi_scale.py           # Main training script for multi-task learning
├── train_transformer_totalscore.py # Baseline training script for total score prediction
└── README.md                      # This file
```

## File Descriptions

### Training Scripts

#### `train_multi_scale.py`
Main training script for multi-task learning models that predict both subscale scores and total scores for depression assessment. Supports:
- **PHQ-8** scale (8 subscales) on E-DAIC dataset
- **HAMD-13** scale (13 subscales) on CIDH and PDCH datasets
- Multiple training strategies: Task-Level SPL, Cluster Constraint, Task Graph GAT
- Regression and ordinal classification modes

#### `train_transformer_totalscore.py`
Baseline training script for predicting total depression scores (single-task regression). Simpler model for comparison purposes.

### Models

#### `models/multi_scale_transformer.py`
Multi-scale Transformer model that:
- Uses BERT embeddings for text encoding
- Applies Transformer layers for sequence modeling
- Predicts multiple subscale scores simultaneously (multi-task learning)
- Supports label normalization and different prediction modes

#### `models/task_graph_gat.py`
Task Graph GAT module that:
- Models relationships between depression subscales using Graph Attention Networks
- Learns task embeddings based on clinical symptom clusters
- Provides gated fusion of shared and task-specific embeddings

### Data Preprocessing

#### `preprocessing/HAMD13Dataset.py`
Dataset loader for HAMD-13 depression assessment:
- Supports **CIDH** and **PDCH** datasets
- Handles both original dialogue format and summarized format
- Pre-computes BERT embeddings with caching
- Splits transcripts into utterances for sequence modeling

#### `preprocessing/EDAICDataset.py`
Dataset loader for PHQ-8 depression assessment:
- Supports **E-DAIC** (Extended Distress Analysis Interview Corpus) dataset
- Loads textual transcripts from CSV files
- Pre-computes BERT embeddings with caching
- Supports sliding window for sequence augmentation

### Utilities

#### `utils/multi_scale_config.py`
Unified configuration system that:
- Defines scale configurations (PHQ-8 and HAMD-13)
- Auto-configures dataset paths and model settings
- Manages normalization parameters
- Provides default hyperparameters for each dataset

#### `utils/cluster_constraint_loss.py`
Cluster constraint loss based on hierarchical clustering of subscale correlations:
- Encourages similar predictions for subscales in the same clinical cluster
- Supports both PHQ-8 and HAMD-13 scales

#### `utils/task_level_spl.py`
Task-Level Self-Paced Learning (SPL) implementation:
- Gradually introduces harder tasks during training
- Adapts curriculum based on prediction difficulty
- Supports multiple pacing functions (linear, log, self-paced, mixture)

#### `utils/ordinal_loss.py`
Ordinal classification loss for depression severity prediction:
- Treats scores as ordered categories
- Uses cumulative logits approach

#### `utils/early_stopping.py`
Early stopping callback to prevent overfitting:
- Monitors validation metrics
- Stops training when no improvement is detected

#### `utils/playground_HAMD13.py` & `utils/playground_PHQ8.py`
Playground scripts for evaluating Large Language Models (LLMs) on depression assessment tasks. These are experimental scripts for zero-shot/few-shot evaluation.

## Usage

### Training Multi-Scale Model (`train_multi_scale.py`)

This script trains multi-task models that predict both subscale scores and total scores.

#### Basic Usage

```bash
# Train on HAMD-13 dataset (CIDH)
python train_multi_scale.py --dataset cidh --scale HAMD-13

# Train on PHQ-8 dataset (E-DAIC)
python train_multi_scale.py --dataset edaic --scale PHQ-8

# Train on PDCH dataset (HAMD-13)
python train_multi_scale.py --dataset pdch --scale HAMD-13
```

#### Advanced Options

```bash
# Specify training hyperparameters
python train_multi_scale.py \
    --dataset pdch \
    --scale HAMD-13 \
    --epochs 80 \
    --batch_size 8 \
    --lr 5e-5

# Use multiple random seeds for reproducibility
python train_multi_scale.py \
    --dataset pdch \
    --seeds "1060,1061,1062,1063,1064"

# Or specify a seed range
python train_multi_scale.py \
    --dataset pdch \
    --seed_start 1060 \
    --seed_end 1065

# Enable advanced training strategies
python train_multi_scale.py \
    --dataset pdch \
    --use_task_spl \              # Enable Task-Level Self-Paced Learning
    --use_cluster_constraint \    # Enable Cluster Constraint Loss
    --use_task_graph              # Enable Task Graph GAT

# Use ordinal classification instead of regression
python train_multi_scale.py \
    --dataset pdch \
    --prediction_mode ordinal

# Predict total score only (instead of subscales)
python train_multi_scale.py \
    --dataset pdch \
    --sum_labels

# Select best model based on different metrics
python train_multi_scale.py \
    --dataset pdch \
    --best_metric test_total_mae  # Options: test_total_mae, test_total_rmse, test_mae, test_rmse

# Specify GPU device
python train_multi_scale.py \
    --dataset pdch \
    --device cuda:0
```

#### All Available Arguments

- `--dataset`: Dataset name (`edaic`, `cidh`, `pdch`)
- `--scale`: Depression scale (`PHQ-8`, `HAMD-13`)
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 10)
- `--lr`: Learning rate (default: 2e-4)
- `--sum_labels`: Use total score instead of subscales (flag)
- `--prediction_mode`: Prediction mode (`regression`, `ordinal`, default: `regression`)
- `--seeds`: Comma-separated seed list or range (e.g., `"1060,1061,1062"` or `"1060-1065"`)
- `--seed_start`: Start seed for range
- `--seed_end`: End seed for range
- `--best_metric`: Metric for selecting best seed (`test_total_mae`, `test_total_rmse`, `test_mae`, `test_rmse`)
- `--use_task_spl`: Enable Task-Level Self-Paced Learning (flag)
- `--use_cluster_constraint`: Enable Cluster Constraint Loss (flag)
- `--use_task_graph`: Enable Task Graph GAT (flag)
- `--device`: Device to use (`cuda:0`, `cuda:1`, `cpu`, default: `cuda`)

### Training Baseline Model (`train_transformer_totalscore.py`)

This script trains a simpler baseline model that predicts only the total depression score.

#### Basic Usage

```bash
# Train on HAMD-13 dataset (CIDH)
python train_transformer_totalscore.py --dataset cidh --scale HAMD-13

# Train on PHQ-8 dataset (E-DAIC)
python train_transformer_totalscore.py --dataset edaic --scale PHQ-8

# Train on PDCH dataset (HAMD-13)
python train_transformer_totalscore.py --dataset pdch --scale HAMD-13
```

#### Advanced Options

```bash
# Specify training hyperparameters
python train_transformer_totalscore.py \
    --dataset pdch \
    --scale HAMD-13 \
    --epochs 80 \
    --batch_size 8 \
    --lr 5e-5

# Use multiple random seeds
python train_transformer_totalscore.py \
    --dataset cidh \
    --seeds "1060,1061,1062"

# Or specify a seed range
python train_transformer_totalscore.py \
    --dataset cidh \
    --seed_start 1060 \
    --seed_end 1065
```

#### All Available Arguments

- `--dataset`: Dataset name (`edaic`, `cidh`, `pdch`)
- `--scale`: Depression scale (`PHQ-8`, `HAMD-13`)
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 10)
- `--lr`: Learning rate (default: 2e-4)
- `--seeds`: Comma-separated seed list (e.g., `"1060,1061,1062"`)
- `--seed_start`: Start seed for range
- `--seed_end`: End seed for range (inclusive)

## Dataset Requirements

### Data Directory Structure

The code expects data files to be organized as follows:

```
data/
├── edaic/                          # E-DAIC dataset (PHQ-8)
│   ├── edaic_labels.csv           # Label file with PHQ-8 scores
│   └── [ParticipantID]_P/         # Participant directories
│       └── [ParticipantID]_Transcript.csv
│
├── cidh/                   # CIDH dataset (HAMD-13)
│   ├── eval_summary_train.json
│   ├── eval_summary_val.json
│   └── eval_summary_test.json
│
└── pdch/                           # PDCH dataset (HAMD-13)
    ├── pdch_original_train.json
    ├── pdch_original_val.json
    ├── pdch_original_test.json
    ├── pdch_summary_train.json
    ├── pdch_summary_val.json
    └── pdch_summary_test.json
```

### Model Requirements

The code uses pre-trained BERT models:
- **For PHQ-8 (E-DAIC)**: `mental/mental-bert-base-uncased` (English)
- **For HAMD-13 (CIDH, PDCH)**: `medbert-base-wwm-chinese` (Chinese Medical BERT)

These models should be available via Hugging Face Transformers or local paths configured in `multi_scale_config.py`.

## Output

Training scripts generate:
- **Model checkpoints**: Saved in `outputs/` directory
- **Training logs**: TensorBoard logs and CSV history files
- **Evaluation metrics**: MAE and RMSE for subscales and total scores

## Notes

- All paths in the code use relative paths (`../data`) for portability
- The code automatically configures dataset-specific settings based on the `--dataset` argument
- For small datasets like PDCH, the configuration automatically adjusts hyperparameters (smaller model, more regularization)
- Embeddings are cached to speed up subsequent runs
EOFREADME
