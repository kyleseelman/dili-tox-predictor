"""
DILI Toxicity Predictor -- Full Training Pipeline

End-to-end pipeline for predicting Drug-Induced Liver Injury using
real FDA datasets and multiple model architectures:

  1. Download real DILI datasets (DILIst, proxy-DILI assays, Cmax PK)
  2. Featurize molecules (Morgan fingerprints, descriptors, graphs)
  3. Train classical models (Random Forest, XGBoost, SVM)
  4. Train feed-forward neural network with multi-modal fusion
  5. Train Graph Neural Network on molecular graphs
  6. Train HepaTox-Fusion: novel multi-task attentive hybrid model
  7. Mechanistic interpretability analysis
  8. Generate Axiom-style visualizations

Data: FDA DILIst (1,111 drugs), proxy-DILI assays (~13,700 compounds),
      clinical Cmax data (~730 compounds)
"""

import sys
import json
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric")
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.data_loading import load_and_merge, prepare_features, split_data, PROXY_SOURCE_MAP
from src.models.classical import train_and_evaluate_classical
from src.models.neural import train_neural_net
from src.models.gnn import train_gnn
from src.models.hybrid import train_hybrid
from src.evaluation import compare_models, compute_metrics, print_metrics
from src.interpret import compute_shap_values, print_top_features, mechanistic_analysis
from src.visualize import generate_all_visualizations
from src.pretrain import (
    load_proxy_pretrain_data,
    pretrain_gnn, finetune_gnn,
    pretrain_hybrid, finetune_hybrid,
)


def main():
    start = time.time()
    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)

    # --- 1. Load Data ---
    print("\n" + "="*70)
    print("  STEP 1: Loading Real DILI Datasets")
    print("="*70)
    df = load_and_merge(force_download=False)
    print(f"  Columns: {list(df.columns)}")
    print(f"  Shape: {df.shape}")

    # --- 2. Featurize ---
    print("\n" + "="*70)
    print("  STEP 2: Molecular Featurization")
    print("="*70)
    features = prepare_features(df, fp_nbits=2048)
    print(f"  Fingerprint matrix: {features['X_fp'].shape}")
    print(f"  Descriptor matrix:  {features['X_desc'].shape}")
    print(f"  Combined features:  {features['X_combined'].shape}")
    print(f"  Labels: {len(features['y'])} "
          f"(DILI+: {int(features['y'].sum())}, "
          f"DILI-: {int((features['y'] == 0).sum())})")

    # --- 3. Split ---
    print("\n" + "="*70)
    print("  STEP 3: Train/Test Split (80/20, stratified)")
    print("="*70)
    split = split_data(features, test_size=0.2)
    print(f"  Train: {len(split['y_train'])} "
          f"(DILI+: {int(split['y_train'].sum())})")
    print(f"  Test:  {len(split['y_test'])} "
          f"(DILI+: {int(split['y_test'].sum())})")

    all_results = {}

    # --- 4. Classical Models ---
    print("\n" + "="*70)
    print("  STEP 4: Training Classical Models (RF, XGBoost, SVM)")
    print("="*70)
    classical_results = train_and_evaluate_classical(
        split["X_train"], split["y_train"],
        split["X_test"], split["y_test"],
    )
    all_results.update(classical_results)

    # --- 5. Neural Network ---
    print("\n" + "="*70)
    print("  STEP 5: Training Multi-Modal Neural Network")
    print("="*70)

    desc_start = features["X_fp"].shape[1]

    X_desc_ext_train = split["X_train"][:, desc_start:]
    X_desc_ext_test = split["X_test"][:, desc_start:]

    nn_result = train_neural_net(
        split["X_fp_train"], X_desc_ext_train, split["y_train"],
        split["X_fp_test"], X_desc_ext_test, split["y_test"],
        epochs=100,
    )
    all_results["NeuralNet"] = nn_result

    # --- 6. Graph Neural Network ---
    print("\n" + "="*70)
    print("  STEP 6: Training Graph Neural Network (GCN)")
    print("="*70)
    gnn_result = train_gnn(
        split["smiles_train"], split["y_train"],
        split["smiles_test"], split["y_test"],
        epochs=80,
    )
    all_results["GNN"] = gnn_result

    # --- 7. HepaTox-Fusion (Novel Hybrid) ---
    print("\n" + "="*70)
    print("  STEP 7: Training HepaTox-Fusion (Multi-Task Attentive Hybrid)")
    print("="*70)

    proxy_candidates = list(PROXY_SOURCE_MAP.keys()) + ["Other_Hepatotox"]
    proxy_cols_present = [c for c in proxy_candidates if c in df.columns]
    valid_idx = features["valid_idx"]
    df_valid = df.iloc[valid_idx].reset_index(drop=True)

    proxy_train, proxy_test = None, None
    if proxy_cols_present:
        proxy_all = df_valid[proxy_cols_present].values.astype(np.float32)
        proxy_all = np.nan_to_num(proxy_all, nan=-1.0)
        proxy_train = proxy_all[split["train_idx"]]
        proxy_test = proxy_all[split["test_idx"]]
        print(f"  Proxy-DILI auxiliary tasks: {proxy_cols_present}")

    hybrid_result = train_hybrid(
        split["smiles_train"], split["y_train"],
        split["smiles_test"], split["y_test"],
        proxy_train=proxy_train,
        proxy_test=proxy_test,
        epochs=120,
        aux_weight=0.3,
    )
    all_results["HepaTox-Fusion"] = hybrid_result

    # --- 8. Pre-train on 8K proxy-DILI → Fine-tune on DILIst ---
    print("\n" + "="*70)
    print("  STEP 8: Pre-train on 8,558 Proxy-DILI Compounds")
    print("="*70)

    proxy_df = load_proxy_pretrain_data()

    print("\n  Pre-training GNN (multi-task) on proxy-DILI...")
    pt_gnn = pretrain_gnn(proxy_df, epochs=50, batch_size=64)

    print("\n  Fine-tuning GNN on DILIst (staged: warmup → full)...")
    ft_gnn_result = finetune_gnn(
        pt_gnn, split["smiles_train"], split["y_train"],
        split["smiles_test"], split["y_test"],
        epochs_warmup=20, epochs_full=50,
    )
    all_results["GNN-PreTrained"] = ft_gnn_result

    print("\n  Pre-training HepaTox-Fusion (multi-task) on proxy-DILI...")
    pt_hybrid = pretrain_hybrid(proxy_df, epochs=50, batch_size=64)

    print("\n  Fine-tuning HepaTox-Fusion on DILIst (staged)...")
    ft_hybrid_result = finetune_hybrid(
        pt_hybrid, split["smiles_train"], split["y_train"],
        split["smiles_test"], split["y_test"],
        proxy_train=proxy_train, proxy_test=proxy_test,
        epochs_warmup=20, epochs_full=60,
    )
    all_results["Fusion-PreTrained"] = ft_hybrid_result

    # --- 9. Compare All Models ---
    print("\n" + "="*70)
    print("  STEP 9: Model Comparison")
    print("="*70)
    summary = compare_models(all_results, split["y_test"])

    # --- 10. Interpretability ---
    print("\n" + "="*70)
    print("  STEP 10: Mechanistic Interpretability Analysis")
    print("="*70)

    best_classical = max(
        [(n, r) for n, r in classical_results.items()],
        key=lambda x: compute_metrics(split["y_test"], x[1]["test_probs"])["AUC-ROC"]
    )
    print(f"  Analyzing best classical model: {best_classical[0]}")

    shap_result = compute_shap_values(
        best_classical[1]["model"],
        split["X_train"],
        split["X_test"],
        features["feature_names"],
        y_test=split["y_test"],
    )
    print_top_features(shap_result, top_n=25)
    mech = mechanistic_analysis(shap_result, top_n=40)

    # --- 11. Visualizations ---
    print("\n" + "="*70)
    print("  STEP 11: Generating Visualizations")
    print("="*70)

    best_model_name = max(summary, key=lambda k: summary[k]["AUC-ROC"])
    print(f"  Best overall model: {best_model_name}")

    viz_paths = generate_all_visualizations(
        results=all_results,
        summary=summary,
        y_test=split["y_test"],
        importance_result=shap_result,
        mechanistic_result=mech,
        desc_matrix_test=split["X_desc_test"],
        desc_names=features["desc_names"],
        smiles_test=split["smiles_test"],
        best_model_name=best_model_name,
        output_dir=output_dir,
    )

    # --- 12. Save Results ---
    elapsed = time.time() - start
    output = {
        "dataset": {
            "total_compounds": len(df),
            "valid_compounds": len(features["y"]),
            "dili_positive": int(features["y"].sum()),
            "dili_negative": int((features["y"] == 0).sum()),
            "num_features": features["X_combined"].shape[1],
            "fp_bits": features["X_fp"].shape[1],
            "num_descriptors": features["X_desc"].shape[1],
            "proxy_dili_tasks": proxy_cols_present,
        },
        "model_performance": {},
        "elapsed_seconds": round(elapsed, 1),
    }
    for name, metrics in summary.items():
        output["model_performance"][name] = {
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in metrics.items()
        }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Results saved to {results_path}")
    print(f"  Visualizations: {[p.name for p in viz_paths]}")
    print(f"  Total pipeline time: {elapsed:.1f}s")
    print("\n  DONE!")


if __name__ == "__main__":
    main()
