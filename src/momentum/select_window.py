"""Select the predictor window for R4/R5 with pre-2023 chronological validation."""

import json

import numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, roc_auc_score
from xgboost import XGBClassifier

from config import GAME_WINDOW_SIZES, PROCESSED_DIR, RANDOM_SEED
from . import classification, regression
from .temporal import date_folds

CUTOFF_YEAR = 2023
N_SPLITS = 5


def evaluate() -> dict:
    results = {}
    observations = None
    for window in GAME_WINDOW_SIZES:
        reg, _ = regression.build_dataset(window=window)
        clf = classification.build_dataset(window=window)
        reg = reg[reg["year"].lt(CUTOFF_YEAR)].sort_values(["date", "match_id", "game_num"])
        clf = clf[clf["year"].lt(CUTOFF_YEAR)].sort_values(["date", "match_id", "game_num"])
        keys = ["date", "match_id", "game_num"]
        reg_keys = reg[keys].reset_index(drop=True)
        if not reg_keys.equals(clf[keys].reset_index(drop=True)):
            raise ValueError("R4 y R5 deben compartir los juegos de validación.")
        if observations is not None and not observations.equals(reg_keys):
            raise ValueError("Todas las ventanas deben evaluarse con los mismos juegos.")
        observations = reg_keys
        reg_cols = regression.get_feature_cols(reg)
        clf_cols = classification.get_feature_cols(clf)
        folds = []
        for train, valid in date_folds(reg["date"], N_SPLITS):
            reg_train, reg_valid = reg.iloc[train].copy(), reg.iloc[valid].copy()
            clf_train, clf_valid = clf.iloc[train].copy(), clf.iloc[valid].copy()
            regression.impute_features(reg_train, reg_valid, reg_cols)
            classification.impute_features(clf_train, clf_valid, clf_cols)
            reg_model = clone(regression.MODELS["XGBoost"])
            reg_model.fit(reg_train[reg_cols], reg_train["N_next"])
            clf_model = XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_SEED,
                n_jobs=1, verbosity=0, eval_metric="logloss",
            )
            clf_model.fit(clf_train[clf_cols], clf_train["next_dir_up"])
            folds.append({
                "train_end": str(reg_train["date"].max().date()),
                "validation_start": str(reg_valid["date"].min().date()),
                "validation_end": str(reg_valid["date"].max().date()),
                "train_games": len(train),
                "validation_games": len(valid),
                "mae": mean_absolute_error(reg_valid["N_next"], reg_model.predict(reg_valid[reg_cols])),
                "auc": roc_auc_score(clf_valid["next_dir_up"], clf_model.predict_proba(clf_valid[clf_cols])[:, 1]),
            })
        results[window] = {
            "mae": float(np.mean([fold["mae"] for fold in folds])),
            "auc": float(np.mean([fold["auc"] for fold in folds])),
            "folds": folds,
        }
    # Common input window: give both endpoints equal weight and break ties by the primary R4 metric.
    mae_order = sorted(results, key=lambda w: results[w]["mae"])
    auc_order = sorted(results, key=lambda w: -results[w]["auc"])
    selected = min(results, key=lambda w: (mae_order.index(w) + auc_order.index(w), mae_order.index(w)))
    return {"cutoff_year": CUTOFF_YEAR, "selected_window": selected, "windows": results}


if __name__ == "__main__":
    result = evaluate()
    output = json.dumps(result, indent=2) + "\n"
    (PROCESSED_DIR / "r4_r5_window_selection.json").write_text(output, encoding="utf-8")
    print(output)
