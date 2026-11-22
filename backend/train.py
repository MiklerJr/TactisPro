"""Entrena y exporta los modelos R4 y R5 para el servicio de inferencia."""

import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBClassifier

from config import PROCESSED_DIR, RANDOM_SEED
from .paths import MODEL_BUNDLE
from src.momentum.classification import (
    build_dataset as build_classification_dataset,
    get_feature_cols as get_classification_features,
)
from src.momentum.regression import (
    MODELS as REGRESSION_MODELS,
    build_dataset as build_regression_dataset,
    get_feature_cols as get_regression_features,
)

TRAIN_CUTOFF_YEAR = 2023


def train_models() -> dict:
    reg_data, _ = build_regression_dataset()
    reg_train = reg_data[reg_data["year"] < TRAIN_CUTOFF_YEAR].copy()
    reg_features = get_regression_features(reg_data)
    reg_medians = reg_train[reg_features].median(numeric_only=True)

    r4 = json.loads((PROCESSED_DIR / "r4_results.json").read_text())
    reg_name = r4["best"]["model"]
    if reg_name == "SVR":
        reg_model = make_pipeline(StandardScaler(), SVR(kernel="rbf", C=1.0))
        rng = np.random.default_rng(RANDOM_SEED)
        reg_train = reg_train.iloc[rng.choice(len(reg_train), size=min(5000, len(reg_train)), replace=False)]
        reg_medians = reg_train[reg_features].median(numeric_only=True)
    else:
        reg_model = clone(REGRESSION_MODELS[reg_name])
    reg_model.fit(reg_train[reg_features].fillna(reg_medians), reg_train["N_next"])

    clf_data = build_classification_dataset()
    clf_train = clf_data[clf_data["year"] < TRAIN_CUTOFF_YEAR].copy()
    clf_features = get_classification_features(clf_data)
    clf_medians = clf_train[clf_features].median(numeric_only=True)
    clf_train.loc[:, clf_features] = clf_train[clf_features].fillna(clf_medians)

    r5 = json.loads((PROCESSED_DIR / "r5_results.json").read_text())
    clf_model = XGBClassifier(
        **r5["gridsearch"]["best_params"], objective="binary:logistic",
        eval_metric="logloss", random_state=RANDOM_SEED, n_jobs=1, verbosity=0,
    )
    clf_model.fit(clf_train[clf_features], clf_train["next_dir_up"].astype(int))

    return {
        "reg_model": reg_model,
        "reg_features": reg_features,
        "reg_medians": reg_medians,
        "clf_model": clf_model,
        "clf_features": clf_features,
        "clf_medians": clf_medians,
        "r4_results": r4,
        "r5_results": r5,
    }


def main() -> None:
    models = train_models()
    with NamedTemporaryFile(dir=MODEL_BUNDLE.parent, suffix=".joblib", delete=False) as file:
        temporary = Path(file.name)
    try:
        joblib.dump(models, temporary, compress=3)
        temporary.replace(MODEL_BUNDLE)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Modelos R4 y R5 exportados a {MODEL_BUNDLE}")


if __name__ == "__main__":
    main()
