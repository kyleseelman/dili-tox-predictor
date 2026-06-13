"""
Classical ML models for DILI prediction.

Implements Random Forest, XGBoost, and SVM classifiers following the
approach in Seal et al. (2024) and the PMC7139829 study, using Morgan
fingerprints + physicochemical descriptors as features.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
import xgboost as xgb


def build_random_forest(n_estimators: int = 500, random_state: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )


def build_xgboost(n_estimators: int = 300, random_state: int = 42, scale_pos_weight: float = 1.0) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )


def build_svm(random_state: int = 42) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=random_state,
        )),
    ])


def train_and_evaluate_classical(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_cv_folds: int = 5,
) -> dict:
    """
    Train RF, XGBoost, SVM and return fitted models with CV + test predictions.
    """
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos = neg / max(pos, 1)

    models = {
        "RandomForest": build_random_forest(),
        "XGBoost": build_xgboost(scale_pos_weight=scale_pos),
        "SVM": build_svm(),
    }

    cv = StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=42)
    results = {}

    for name, model in models.items():
        print(f"  Training {name}...")

        cv_probs = cross_val_predict(
            model, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
        )[:, 1]

        model.fit(X_train, y_train)

        test_probs = model.predict_proba(X_test)[:, 1]
        test_preds = model.predict(X_test)

        results[name] = {
            "model": model,
            "cv_probs": cv_probs,
            "test_probs": test_probs,
            "test_preds": test_preds,
        }

    return results
