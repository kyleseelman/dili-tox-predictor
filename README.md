# DILI Toxicity Predictor

Multi-modal machine learning pipeline for predicting **Drug-Induced Liver Injury (DILI)** from chemical structure, in vitro assay data, and pharmacokinetic parameters. DILI is responsible for ~20-25% of clinical trial failures and is the leading cause of post-market drug withdrawals.

## Visualizations

### Model Performance Comparison
<p align="center"><img src="outputs/model_comparison.png" width="800"/></p>

### ROC & Precision-Recall Curves
<p align="center"><img src="outputs/roc_pr_curves.png" width="800"/></p>

### Mechanistic Radar — DILI Risk Factor Decomposition
<p align="center"><img src="outputs/mechanistic_radar.png" width="500"/></p>

### Top Features Driving Hepatotoxicity Predictions
<p align="center"><img src="outputs/feature_importance.png" width="700"/></p>

### Compound Risk Comparison (Advance vs Stop)
<p align="center"><img src="outputs/compound_risk_comparison.png" width="800"/></p>

### Chemical Space — Toxicity Landscape (t-SNE)
<p align="center"><img src="outputs/toxicity_landscape.png" width="800"/></p>

### Calibration & Probability Distribution
<p align="center"><img src="outputs/calibration_distribution.png" width="800"/></p>

## Data Sources

All data is from peer-reviewed, publicly available sources — no synthetic or mock data:

| Dataset | Source | Compounds | Description |
|---------|--------|-----------|-------------|
| **DILIst Gold Standard** | [FDA LTKB](https://www.fda.gov/science-research/liver-toxicity-knowledge-base-ltkb) / [Seal et al. 2024](https://github.com/srijitseal/DILI) | 1,111 | Binary DILI classification of FDA-approved drugs |
| **Proxy-DILI Labels** | Seal et al. 2024 | 13,703 | Mitochondrial toxicity, BSEP inhibition, rat hepatotoxicity |
| **Cmax Pharmacokinetics** | Seal et al. 2024 | 727 | Clinical plasma concentration (total and unbound) |

## Models

Six architectures trained and compared:

| Model | Architecture | AUC-ROC | AUC-PR | Sensitivity | Specificity | MCC |
|-------|-------------|---------|--------|-------------|-------------|-----|
| GNN | Graph Convolutional Network on molecular graphs | **0.708** | **0.781** | 0.535 | 0.731 | 0.256 |
| HepaTox-Fusion | Multi-task GIN + highway FP + cross-attention *(novel)* | 0.674 | 0.748 | 0.563 | 0.680 | 0.233 |
| NeuralNet | Multi-modal feed-forward (FP + descriptors + assays) | 0.689 | 0.760 | 0.754 | 0.577 | **0.329** |
| Random Forest | 500 trees on Morgan FP + descriptors + assays | 0.675 | 0.767 | 0.838 | 0.359 | 0.223 |
| XGBoost | Gradient-boosted trees | 0.618 | 0.748 | 0.718 | 0.449 | 0.169 |
| SVM | RBF kernel with balanced classes | 0.611 | 0.733 | 0.951 | 0.154 | 0.178 |

## HepaTox-Fusion Architecture

A **Multi-Task Attentive Fusion Network** inspired by recent research:

- [DeepGPT-DILI (2025)](https://hub.tmu.edu.tw/en/publications/deepgpt-dili-integrating-graph-convolutional-networks-andlarge-la/): Ensemble of GCN + fingerprints + LLM embeddings
- [DILIGeNN (2025)](https://link.springer.com/article/10.1186/s13321-025-01068-3): GNN with augmented graph features
- [Multi-task toxicity learning (2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12912713/): Shared representations across correlated endpoints

```
SMILES ──► GIN (4-layer Graph Isomorphism Network) ──► graph_emb (128d)
       ──► Highway FP (3-layer gated network on Morgan FP) ──► fp_emb (256d)
       ──► Cross-Attention (FP queries attend to GIN nodes) ──► fused (256d)
       ──► Descriptor Branch (physicochemical props) ──► desc_emb (32d)
                          │
                ┌─────────┴─────────┐
                │  Shared Trunk     │
                ├───────────────────┤
                │ Primary: DILI     │
                │ Aux: Mito Tox     │
                │ Aux: BSEP         │
                │ Aux: Rat Hepato   │
                │ Aux: ...          │
                └───────────────────┘
```

**Key ideas:** Cross-attention between FP embeddings and graph node embeddings identifies which substructures drive toxicity. Multi-task learning with proxy-DILI auxiliary objectives regularizes training on the small 1,111-compound dataset.

## Features

### Molecular Featurization
- **Morgan fingerprints** (2048-bit, radius 2)
- **20 physicochemical descriptors**: MolWt, LogP, TPSA, HBD/HBA, aromatic rings, FractionCSP3, partial charges, etc.
- **5 Lipinski Rule-of-5 alerts**
- **Graph representation**: Atom features (139-dim) + bond features (7-dim)

### Proxy-DILI Assay Labels
- Mitochondrial toxicity, BSEP inhibition, rat hepatotoxicity, and other proxy endpoints merged from 13,703 compounds

### Mechanistic Interpretability
Feature importance decomposed into hepatotoxicity mechanisms: lipophilicity (steatosis), structural complexity (reactive metabolites), polarity (cholestasis), reactivity (ER stress/ROS), and in vitro assay signals.

## Project Structure

```
dili-tox-predictor/
├── train.py                        # Full training pipeline
├── data/
│   ├── download.py                 # Auto-downloads real datasets
│   └── *.csv                       # Downloaded on first run
├── src/
│   ├── data_loading.py             # Data merging and preprocessing
│   ├── featurization.py            # Molecular featurization
│   ├── evaluation.py               # Metrics and model comparison
│   ├── interpret.py                # Feature importance analysis
│   ├── visualize.py                # Publication-quality plots
│   └── models/
│       ├── classical.py            # Random Forest, XGBoost, SVM
│       ├── neural.py               # Feed-forward neural network
│       ├── gnn.py                  # Graph Convolutional Network
│       └── hybrid.py               # HepaTox-Fusion
└── outputs/                        # Generated results and plots
```

## Quick Start

```bash
cd dili-tox-predictor
python train.py
```

Datasets are automatically downloaded on first run. The full pipeline trains 6 models and generates all visualizations.

## References

- Seal, S. et al. (2024). *Enhancing Early Detection of Drug-Induced Liver Injury*. [GitHub](https://github.com/srijitseal/DILI)
- Chen, M. et al. (2016). *DILIrank: the largest reference drug list ranked by DILI risk*. Drug Discovery Today.
- Thakkar, S. et al. (2020). *DILIst: binary classification of 1279 drugs*. Drug Discovery Today.
- DeepGPT-DILI (2025). *Integrating GCN and LLM Embeddings for DILI Prediction*.
- DILIGeNN (2025). *GNNs with augmented graph features*. J. Cheminformatics.
- HTP-Pred (2025). *GNN-based hepatotoxicity prediction*. J. Cheminformatics.
