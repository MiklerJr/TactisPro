"""Inferencia sobre partidos históricos y explicaciones por juego."""

import json
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from xgboost import DMatrix

from config import GAME_FEATURES_RESOLVED_DATASET, MOMENTUM_DATASET, PROCESSED_DIR
from src.momentum.classification import add_momentum_features
from src.momentum.classification import build_dataset as build_classification_dataset
from src.momentum.crossings import momentum_crossings
from src.momentum.regression import build_dataset as build_regression_dataset
from .paths import MODEL_BUNDLE


def records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records", double_precision=15))


def compute_predictions(reg: pd.DataFrame, clf: pd.DataFrame, models: dict) -> pd.DataFrame:
    """Genera predicciones y errores por partido para evaluación fuera del replay."""
    reg_features = models["reg_features"]
    clf_features = models["clf_features"]

    reg_x = reg[reg_features].fillna(models["reg_medians"])
    clf_x = clf[clf_features].fillna(models["clf_medians"])

    frame = reg[["match_id", "game_num", "year", "N_t", "N_next"]].copy()
    frame = frame.rename(columns={"N_next": "N_observed_next"})
    frame["N_pred_next"] = models["reg_model"].predict(reg_x)
    frame["error_abs"] = (frame["N_pred_next"] - frame["N_observed_next"]).abs()

    prob = models["clf_model"].predict_proba(clf_x)[:, 1]
    clf_small = clf[["game_num", "next_dir_up"]].copy()
    clf_small["prob_up"] = prob
    clf_small["dir_pred"] = np.where(prob >= 0.5, "Aumenta", "No aumenta")
    clf_small["dir_observed"] = np.where(
        clf_small["next_dir_up"] == 1, "Aumenta", "No aumenta"
    )
    clf_small["dir_correct"] = clf_small["dir_pred"] == clf_small["dir_observed"]

    frame = frame.merge(
        clf_small[["game_num", "prob_up", "dir_pred", "dir_observed", "dir_correct"]],
        on="game_num",
        how="inner",
    )
    return frame.reset_index(drop=True)



class ReplayInference:
    """Carga los artefactos locales y prepara los datasets una vez por proceso."""

    def __init__(self) -> None:
        if not MODEL_BUNDLE.is_file():
            raise FileNotFoundError(f"{MODEL_BUNDLE}. Ejecuta python -m backend.train")
        self.models = joblib.load(MODEL_BUNDLE)
        self.gru = None
        if (PROCESSED_DIR / "r6_gru.pt").is_file():
            # Load the XGBoost bundle before importing PyTorch on macOS.
            import torch
            from src.momentum.crossing_forecast import CrossingGRU, MODEL_PATH

            self.gru = CrossingGRU()
            self.gru.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
            self.gru.eval()

    def metrics(self) -> dict:
        return {
            "r4": self.models["r4_results"]["best"],
            "r5": self.models["r5_results"]["gridsearch"]["test_metrics"],
            "r6_available": self.gru is not None,
        }

    def game(self, match_id: str, game_num: int, timings: dict[str, float] | None = None) -> dict:
        """Ejecuta la inferencia para un juego cerrado con entradas hasta ese juego."""
        if game_num < 1:
            raise KeyError((match_id, game_num))
        from src.momentum.crossing_forecast import HISTORY, INPUTS

        started = perf_counter()
        momentum = pd.read_parquet(
            MOMENTUM_DATASET, columns=["game_num", *INPUTS],
            filters=[("match_id", "=", match_id), ("game_num", "<=", game_num),
                     ("game_num", ">", max(0, game_num - HISTORY))],
        ).sort_values("game_num").set_index("game_num")
        features = pd.read_parquet(
            GAME_FEATURES_RESOLVED_DATASET,
            filters=[("match_id", "=", match_id), ("game_num", "=", game_num),
                     ("window_size", "=", 5)],
        )
        if game_num not in momentum.index or len(features) != 1:
            raise KeyError((match_id, game_num))
        loaded = perf_counter()

        current = momentum.loc[game_num]
        previous = momentum.reindex([game_num - 1]).iloc[0]
        for column in INPUTS:
            features[f"{column}_t"] = current[column]
            features[f"{column}_prev"] = previous[column]
        add_momentum_features(features)

        models = self.models
        reg_x = features[models["reg_features"]].fillna(models["reg_medians"])
        clf_x = features[models["clf_features"]].fillna(models["clf_medians"])
        prepared = perf_counter()
        regression = float(models["reg_model"].predict(reg_x)[0])
        probability = float(models["clf_model"].predict_proba(clf_x)[0, 1])
        predicted = perf_counter()
        shap = models["clf_model"].get_booster().predict(DMatrix(clf_x), pred_contribs=True)[0, :-1]
        explained = perf_counter()
        horizon = None
        if self.gru is not None and momentum.index.tolist() == list(range(game_num - HISTORY + 1, game_num + 1)):
            import torch

            values = momentum[INPUTS].to_numpy(dtype=np.float32)
            with torch.inference_mode():
                horizon = float(self.gru(torch.from_numpy(values).unsqueeze(0)).item())

        if timings is not None:
            timings.update(
                lectura=(loaded - started) * 1000,
                preparacion=(prepared - loaded) * 1000,
                inferencia=(predicted - prepared + perf_counter() - explained) * 1000,
                shap=(explained - predicted) * 1000,
            )
        return {
            "prediction": {
                "game_num": game_num, "N_t": float(current["N"]), "N_pred_next": regression,
                "prob_up": probability, "dir_pred": "Aumenta" if probability >= 0.5 else "No aumenta",
                "games_to_crossing": horizon,
            },
            "contributions": dict(zip(models["clf_features"], map(float, shap))),
        }

    def match(self, match_id: str) -> dict:
        """Genera evidencia por partido para los notebooks de evaluación."""
        reg, _ = build_regression_dataset(match_id=match_id)
        clf = build_classification_dataset(match_id=match_id)
        momentum = pd.read_parquet(
            MOMENTUM_DATASET, filters=[("match_id", "=", match_id)],
        ).sort_values("game_num").set_index("game_num")
        predictions = compute_predictions(reg, clf, self.models)
        columns = self.models["clf_features"]
        features = clf[columns].fillna(self.models["clf_medians"])
        shap_values = self.models["clf_model"].get_booster().predict(
            DMatrix(features), pred_contribs=True,
        )
        shap = pd.DataFrame(shap_values[:, :-1], columns=columns)
        shap.insert(0, "game_num", clf["game_num"].to_numpy())

        if self.gru is not None:
            import torch
            from src.momentum.crossing_forecast import HISTORY, INPUTS

        if self.gru is not None and len(momentum) >= HISTORY:
            values = momentum[INPUTS].to_numpy(dtype=np.float32)
            windows = np.stack([values[end-HISTORY:end] for end in range(HISTORY, len(values)+1)])
            with torch.inference_mode():
                forecasts = self.gru(torch.from_numpy(windows)).numpy()
            horizons = pd.Series(forecasts, index=momentum.index[HISTORY-1:])
            predictions["games_to_crossing"] = predictions["game_num"].map(horizons)
        else:
            predictions["games_to_crossing"] = np.nan

        crossings = momentum_crossings(momentum["N"]).rename("direction").reset_index()
        return {
            "predictions": records(predictions),
            "contributions": records(shap),
            "crossings": records(crossings),
        }
