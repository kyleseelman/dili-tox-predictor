"""
Model interpretability for DILI predictions.

Implements feature importance analysis to identify which molecular
properties and structural features drive hepatotoxicity predictions
-- mirroring Axiom Bio's emphasis on mechanistic understanding and
structure-toxicity relationships.

Uses tree-based feature importances (MDI / permutation) and, when
available, SHAP TreeExplainer for per-sample attribution.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance


def compute_shap_values(
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str],
    y_test: np.ndarray | None = None,
    max_train_samples: int = 200,
) -> dict:
    """
    Compute feature importances using the best available method:
    1. SHAP TreeExplainer (if shap works)
    2. MDI (built-in feature_importances_ for tree models)
    3. Permutation importance as fallback
    """
    shap_values = None

    try:
        import shap
        explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        shap_values = explainer.shap_values(X_test)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        print("  Using SHAP TreeExplainer for interpretability")
    except Exception:
        if hasattr(model, "feature_importances_"):
            mean_abs_shap = model.feature_importances_
            print("  Using MDI (Gini) feature importance")
        elif y_test is not None:
            perm = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
            mean_abs_shap = perm.importances_mean
            print("  Using permutation importance")
        else:
            mean_abs_shap = np.zeros(len(feature_names))
            print("  Warning: could not compute feature importance")

    top_idx = np.argsort(mean_abs_shap)[::-1]

    return {
        "shap_values": shap_values,
        "mean_abs_shap": mean_abs_shap,
        "top_feature_idx": top_idx,
        "feature_names": feature_names,
    }


def print_top_features(shap_result: dict, top_n: int = 20):
    """Print top features driving DILI predictions."""
    print(f"\nTop {top_n} Features Driving Hepatotoxicity Predictions:")
    print("-" * 55)
    for rank, idx in enumerate(shap_result["top_feature_idx"][:top_n]):
        name = shap_result["feature_names"][idx]
        importance = shap_result["mean_abs_shap"][idx]
        print(f"  {rank+1:>3}. {name:<35} {importance:.6f}")
    print()


def mechanistic_analysis(shap_result: dict, top_n: int = 30) -> dict:
    """
    Group top features into mechanistic categories relevant to DILI.

    Categories aligned with Axiom Bio's hepatotoxicity mechanisms:
    - Lipophilicity: LogP, lipophilic descriptors (-> steatosis, membrane disruption)
    - Structural complexity: MolWt, ring count, aromatic rings (-> reactive metabolites)
    - Polarity: TPSA, HBD/HBA (-> BSEP inhibition, cholestasis)
    - Reactivity: charge, valence electrons (-> reactive metabolite formation)
    - Assay readouts: proxy-DILI labels (-> in vitro mechanism signals)
    """
    categories = {
        "Lipophilicity (Steatosis risk)": ["LogP", "MolLogP", "FractionCSP3", "Ro5_LogP"],
        "Structural Complexity (Metabolite risk)": [
            "MolWt", "NumAromaticRings", "RingCount", "NumHeavyAtoms",
            "HeavyAtomMolWt", "Ro5_MolWt",
        ],
        "Polarity (Cholestasis risk)": [
            "TPSA", "NumHDonors", "NumHAcceptors", "Ro5_HBD", "Ro5_HBA",
        ],
        "Molecular Shape": [
            "NumRotatableBonds", "LabuteASA", "MolMR", "BalabanJ",
            "NumAliphaticRings", "NumSaturatedRings",
        ],
        "Reactivity (ER stress / ROS risk)": [
            "MaxPartialCharge", "MinPartialCharge", "NumValenceElectrons",
        ],
        "In Vitro Assays": [
            "Mito_Tox", "BSEP_inhibition", "BSEP", "Rat_Hepatotox",
            "THLE_cytotox", "HepG2_cytotox", "h_Hepatocyte_cytotox",
            "reactive_metabolite", "Pgp_inhibition",
        ],
        "Pharmacokinetics": ["Cmax_uM", "Total_Cmax_uM"],
    }

    names = shap_result["feature_names"]
    importances = shap_result["mean_abs_shap"]
    top_features = set(
        names[i] for i in shap_result["top_feature_idx"][:top_n]
        if not names[i].startswith("FP_")
    )

    results = {}
    print("\nMechanistic Decomposition of DILI Risk Factors:")
    print("=" * 60)

    for category, keywords in categories.items():
        matching = []
        for feat_name in top_features:
            if any(kw.lower() in feat_name.lower() for kw in keywords):
                idx = names.index(feat_name) if feat_name in names else -1
                if idx >= 0:
                    matching.append((feat_name, importances[idx]))

        if matching:
            matching.sort(key=lambda x: x[1], reverse=True)
            total_importance = sum(m[1] for m in matching)
            results[category] = {
                "features": matching,
                "total_importance": total_importance,
            }
            print(f"\n  {category}")
            print(f"  {'─' * 50}")
            for feat, imp in matching:
                print(f"    {feat:<35} {imp:.6f}")
            print(f"    {'Total:':<35} {total_importance:.6f}")

    fp_importance = sum(
        importances[i] for i in range(len(names)) if names[i].startswith("FP_")
    )
    print(f"\n  Structural Fingerprints (substructure patterns)")
    print(f"  {'─' * 50}")
    print(f"    {'Total FP importance:':<35} {fp_importance:.6f}")

    print("=" * 60)
    return results
