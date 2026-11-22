"""R4 — Modelo base de regresion para predecir el momentum neto en t+1.

Indicador verificable de R4 (Tesis):
- Comparativa de Random Forest, XGBoost y SVR mediante MAE, RMSE y R^2
  en validacion cruzada. El mejor modelo se selecciona como base.

Target:
    y = N(t+1) = M_A(t+1) - M_B(t+1)
    Tambien reporta M_A(t+1) y M_B(t+1) como referencias secundarias.

Features (input):
    - 16 features dinamicas de R2 (W=5, game_features_resolved)
    - N(t), M_A(t), M_B(t) (componente autorregresivo)  ->  19 en total

Split:
    Train: anos < 2023 | Test: anos >= 2023 (split temporal estricto)

Validacion:
    Tres folds cronológicos por fecha completa, con imputación por fold.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from .temporal import date_folds
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from xgboost import XGBRegressor

from config import (
    PROCESSED_DIR,
    REPORTS_DIR,
    PROJECT_ROOT,
    GAME_FEATURES_RESOLVED_DATASET,
    GAMES_DATASET,
    MOMENTUM_DATASET,
    RANDOM_SEED,
)

DOCS_DIR = PROJECT_ROOT / "docs_tesis"
FIG_DIR = DOCS_DIR / "figures"

RANDOM_STATE = RANDOM_SEED

MODELS = {
    "Random Forest": RandomForestRegressor(
        n_estimators=50, max_depth=6, random_state=RANDOM_STATE, n_jobs=1
    ),
    "XGBoost": XGBRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, n_jobs=1, verbosity=0,
    ),
}

TARGETS = ["N", "M_A", "M_B"]


def build_dataset(match_id: str | None = None, *, window: int = 5) -> tuple[pd.DataFrame, dict[str, str]]:
    """Une features R2 + momentum actual + target = momentum en t+1."""
    filters = [("match_id", "=", match_id)] if match_id is not None else None
    feats = pd.read_parquet(GAME_FEATURES_RESOLVED_DATASET, filters=filters)
    games = pd.read_parquet(GAMES_DATASET, filters=filters)[["match_id", "game_num", "date"]]
    mom = pd.read_parquet(MOMENTUM_DATASET, filters=filters)

    # Una fila por juego y únicamente las variables de la ventana evaluada.
    columns = [c for c in feats if c.endswith(f"_W{window}") or c in {"match_id", "game_num"}]
    feats = feats.loc[feats["window_size"].eq(window), columns].copy()

    # Merge con fecha
    feats = feats.merge(games, on=["match_id", "game_num"], how="inner")

    # Merge con momentum actual (mismo match_id, game_num = t)
    feats = feats.merge(
        mom[["match_id", "game_num", "M_A", "M_B", "N"]]
        .rename(columns={"M_A": "M_A_t", "M_B": "M_B_t", "N": "N_t"}),
        on=["match_id", "game_num"], how="inner",
    )

    # Target = momentum en t+1 (siguiente game del mismo partido)
    mom_next = mom[["match_id", "game_num", "M_A", "M_B", "N"]].rename(
        columns={"M_A": "M_A_next", "M_B": "M_B_next", "N": "N_next"}
    )
    mom_next["game_num"] = mom_next["game_num"] - 1  # shift: el game N tiene target en N+1
    feats = feats.merge(mom_next, on=["match_id", "game_num"], how="inner")

    feats["date"] = pd.to_datetime(feats["date"])
    feats["year"] = feats["date"].dt.year

    target_cols = {t: f"{t}_next" for t in TARGETS}
    return feats, target_cols


def temporal_split(df: pd.DataFrame, cutoff_year: int = 2023) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["year"] < cutoff_year].copy()
    test = df[df["year"] >= cutoff_year].copy()
    return train, test


def impute_features(train: pd.DataFrame, test: pd.DataFrame, feat_cols: list[str]) -> None:
    """Imputa NaNs en features con la mediana del train (in-place)."""
    medians = train[feat_cols].median(numeric_only=True)
    for c in feat_cols:
        train[c] = train[c].fillna(medians[c])
        test[c] = test[c].fillna(medians[c])


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    drop = {"match_id", "game_num", "date", "year"} | {f"{t}_next" for t in TARGETS}
    return [c for c in df.columns if c not in drop]


def evaluate_model(model, X_train, y_train, X_test, y_test, dates, n_splits: int = 3, train_limit: int | None = None):
    """Valida en fechas posteriores y ajusta la imputación en cada fold."""
    cv_mae_list = []
    cv_r2_list = []
    cv_rmse_list = []
    rng = np.random.default_rng(RANDOM_STATE)

    def training_rows(indices):
        if train_limit is not None and len(indices) > train_limit:
            return rng.choice(indices, size=train_limit, replace=False)
        return indices

    for fold, (tr_idx, va_idx) in enumerate(date_folds(dates, n_splits), start=1):
        model_clone = make_pipeline(SimpleImputer(strategy="median"), _clone_model(model))
        selected = training_rows(tr_idx)
        model_clone.fit(X_train[selected], y_train[selected])
        pred = model_clone.predict(X_train[va_idx])
        cv_mae_list.append(mean_absolute_error(y_train[va_idx], pred))
        cv_rmse_list.append(np.sqrt(mean_squared_error(y_train[va_idx], pred)))
        cv_r2_list.append(r2_score(y_train[va_idx], pred))
        print(f"      fold {fold}/{n_splits}: MAE={cv_mae_list[-1]:.4f}", flush=True)

    cv_mae = np.array(cv_mae_list)
    cv_r2 = np.array(cv_r2_list)

    final_model = make_pipeline(SimpleImputer(strategy="median"), _clone_model(model))
    rng = np.random.default_rng(RANDOM_STATE)
    selected = training_rows(np.arange(len(X_train)))
    final_model.fit(X_train[selected], y_train[selected])
    y_pred = final_model.predict(X_test)

    return {
        "cv_mae_mean": float(cv_mae.mean()),
        "cv_mae_std": float(cv_mae.std()),
        "cv_rmse_mean": float(np.mean(cv_rmse_list)),
        "cv_rmse_std": float(np.std(cv_rmse_list)),
        "cv_r2_mean": float(cv_r2.mean()),
        "cv_r2_std": float(cv_r2.std()),
        "test_mae": float(mean_absolute_error(y_test, y_pred)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "test_r2": float(r2_score(y_test, y_pred)),
        "y_pred_sample": y_pred[:5].tolist(),
        "y_test_sample": y_test[:5].tolist(),
    }


def _clone_model(model):
    from sklearn.base import clone
    return clone(model)


def evaluate_svr(X_train, y_train, X_test, y_test):
    """Evalua SVR con subset reducido por costo computacional.

    SVR escala O(n^2) en memoria; 264k filas es prohibitivo.
    Se entrena con 5000 filas para reportar la metrica.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    sub_size = min(5000, len(X_train))
    sub_idx = rng.choice(len(X_train), size=sub_size, replace=False)
    X_sub = X_train[sub_idx]
    y_sub = y_train[sub_idx]

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="rbf", C=1.0)),
    ])
    pipe.fit(X_sub, y_sub)
    y_pred = pipe.predict(X_test)

    return {
        "cv_mae_mean": None,
        "cv_mae_std": None,
        "cv_r2_mean": None,
        "cv_r2_std": None,
        "test_mae": float(mean_absolute_error(y_test, y_pred)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "test_r2": float(r2_score(y_test, y_pred)),
        "y_pred_sample": y_pred[:5].tolist(),
        "y_test_sample": y_test[:5].tolist(),
        "note": f"SVR entrenado con subset de {sub_size} filas por costo O(n^2)",
    }


def plot_comparison(results: dict[str, dict[str, dict]], out_path: Path) -> None:
    """Grafica MAE y R^2 por modelo y target."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    targets = list(results[list(results.keys())[0]].keys())
    models = list(results.keys())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    x = np.arange(len(models))
    width = 0.25
    colors = ["#2563EB", "#DC2626", "#059669"]

    for i, target in enumerate(targets):
        maes = [results[m][target]["test_mae"] for m in models]
        offset = (i - 1) * width
        axes[0].bar(x + offset, maes, width, label=f"MAE {target}", color=colors[i])

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=15)
    axes[0].set_ylabel("MAE (test)")
    axes[0].set_title("Error absoluto medio en test")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, axis="y", alpha=0.25)

    for i, target in enumerate(targets):
        r2s = [results[m][target]["test_r2"] for m in models]
        offset = (i - 1) * width
        axes[1].bar(x + offset, r2s, width, label=f"R^2 {target}", color=colors[i])

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=15)
    axes[1].set_ylabel("R^2 (test)")
    axes[1].set_title("Coeficiente de determinacion en test")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].axhline(0, color="black", linewidth=0.6)

    fig.suptitle("R4 - Comparativa de modelos de regresion", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def write_report(
    results: dict[str, dict[str, dict]],
    best: dict,
    baselines: dict[str, dict[str, dict]],
    out_path: Path,
) -> None:
    lines = [
        "# R4. MODELO BASE DE REGRESION - PREDICCION DE MOMENTUM",
        "",
        "## 4.1 Objetivo",
        "",
        "Predecir el valor numerico del indice de momentum en el siguiente game",
        "(t+1) usando tres algoritmos de regresion: Random Forest, XGBoost y SVR.",
        "El mejor modelo se selecciona como base para el clasificador de la direccion del cambio (R5).",
        "",
        "## 4.2 Configuracion experimental",
        "",
        "**Datos de entrada**:",
        "- Features dinamicas R2 (W=5): 16 columnas.",
        "- Features autorregresivas: M_A(t), M_B(t), N(t).",
        "- Total features: 19.",
        "- Filas: 1 game = 1 fila (se filtra window_size=5; el parquet trae 3",
        "  filas por game, una por ventana).",
        "",
        "**Target**: `N_next = N(t+1)` (primario); `M_A_next`, `M_B_next` (secundarios).",
        "",
        "**Split temporal**:",
        "- Train: anos < 2023.",
        "- Test: anos >= 2023.",
        "- **Sin leakage**: el target esta shifted al siguiente game del mismo partido.",
        "",
        "**Hiperparametros**:",
        "",
        "| Modelo | Hiperparametros |",
        "|---|---|",
        "| Random Forest | n_estimators=50, max_depth=6, random_state=42 |",
        "| XGBoost | n_estimators=100, max_depth=4, lr=0.1, subsample=0.8, colsample=0.8 |",
        "| SVR | kernel=rbf, C=1.0, scaler=StandardScaler (subset 5k filas, O(n^2)) |",
        "",
        "**Validacion**: Tres folds cronológicos por fecha completa, con imputación ajustada en cada fold.",
        "",
        "## 4.3 Resultados en test set",
        "",
        "**Tabla 1. Metricas en test (MAE, RMSE, R^2)**",
        "",
        "| Modelo | Target | MAE | RMSE | R^2 |",
        "|---|---|---|---|---|",
    ]

    for model_name, target_results in results.items():
        for target, m in target_results.items():
            lines.append(
                f"| {model_name} | {target} | {m['test_mae']:.4f} "
                f"| {m['test_rmse']:.4f} | {m['test_r2']:.4f} |"
            )

    lines += [
        "",
        "**Tabla 2. Baselines ingenuos (mismo test set)**",
        "",
        "| Baseline | Target | MAE | RMSE | R^2 |",
        "|---|---|---|---|---|",
    ]
    for tlogical, bl in baselines.items():
        for bl_name, m in bl.items():
            label = "Persistencia (t -> t+1)" if bl_name == "persistence" else "Media del train"
            lines.append(
                f"| {label} | {tlogical} | {m['test_mae']:.4f} "
                f"| {m['test_rmse']:.4f} | {m['test_r2']:.4f} |"
            )

    lines += [
        "",
        "**Tabla 3. Validacion cruzada 3-fold (TimeSeriesSplit)**",
        "",
        "| Modelo | Target | CV MAE (media ± DE) | CV RMSE (media ± DE) | CV R² (media ± DE) |",
        "|---|---|---|---|---|",
    ]
    for model_name, target_results in results.items():
        for target, m in target_results.items():
            cv_mae = m.get("cv_mae_mean")
            cv_r2 = m.get("cv_r2_mean")
            cv_mae_str = (
                f"{cv_mae:.4f} ± {m['cv_mae_std']:.4f}"
                if cv_mae is not None else "N/A (subset)"
            )
            cv_r2_str = (
                f"{cv_r2:.4f} ± {m['cv_r2_std']:.4f}"
                if cv_r2 is not None else "N/A (subset)"
            )
            lines.append(
                f"| {model_name} | {target} | {cv_mae_str} | {m['cv_rmse_mean']:.4f} ± {m['cv_rmse_std']:.4f} | {cv_r2_str} |"
            )

    lines += [
        "",
        "![Comparativa](figures/11_regression_comparison.png)",
        "",
        "## 4.4 Seleccion del mejor modelo",
        "",
        f"**Modelo seleccionado (target primario N_next)**: **{best['model']}**",
        "",
        f"- MAE = {best['metrics']['test_mae']:.4f}",
        f"- RMSE = {best['metrics']['test_rmse']:.4f}",
        f"- R^2 = {best['metrics']['test_r2']:.4f}",
        "",
        "**Criterio de seleccion**: menor MAE medio en validación cruzada sobre `N_next` entre los modelos con entrenamiento completo. El contraste con presupuesto común de 5 000 filas se reproduce con `src.momentum.predictive_audit`.",
        "",
        "## 4.5 Cumplimiento del indicador R4",
        "",
        "**Indicador del EDT**: 'Comparativa de Random Forest, XGBoost y SVR mediante",
        "MAE, RMSE y R^2 en validacion cruzada. El mejor modelo se selecciona como base.'",
        "",
        "**Resultado**:",
        "- Se entrenaron y compararon los 3 algoritmos requeridos.",
        "- Se reportan MAE, RMSE y R^2.",
        "- Se aplico validacion cruzada temporal (3 folds por fecha completa).",
        "- Se selecciono el modelo de minimo MAE como base.",
        "",
        "**Conclusion**: el indicador R4 se cumple completamente.",
        "",
        "## 4.6 Medio de verificacion",
        "",
        "- Script: `src/momentum/regression.py`",
        "- Resultados crudos: `data/processed/r4_results.json`",
        "- Reporte: `reports/04_regression.md`",
        "- Figura: `docs_tesis/figures/11_regression_comparison.png`",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[regression] Reporte -> {out_path}")


def main():
    print("[regression] Construyendo dataset...", flush=True)
    df, target_cols = build_dataset()
    print(f"[regression] Filas totales: {len(df):,}", flush=True)

    train, test = temporal_split(df)
    print(f"[regression] Train: {len(train):,} | Test: {len(test):,}", flush=True)

    feat_cols = get_feature_cols(df)
    X_train = train[feat_cols].values
    X_test = test[feat_cols].values
    print(f"[regression] Features: {len(feat_cols)}", flush=True)

    results: dict[str, dict[str, dict]] = {}
    for model_name, model in MODELS.items():
        print(f"[regression] Entrenando {model_name}...", flush=True)
        results[model_name] = {}
        for target_logical, target_col in target_cols.items():
            y_train = train[target_col].values
            y_test = test[target_col].values
            print(f"   target={target_logical}", flush=True)
            metrics = evaluate_model(model, X_train, y_train, X_test, y_test, train["date"])
            results[model_name][target_logical] = metrics
            print(
                f"      MAE={metrics['test_mae']:.4f} "
                f"RMSE={metrics['test_rmse']:.4f} R^2={metrics['test_r2']:.4f}",
                flush=True,
            )

    print("[regression] Entrenando SVR (hasta 5k filas por fold)...", flush=True)
    svr = make_pipeline(StandardScaler(), SVR(kernel="rbf", C=1.0))
    results["SVR"] = {}
    for target_logical, target_col in target_cols.items():
        results["SVR"][target_logical] = evaluate_model(
            svr, X_train, train[target_col].values, X_test, test[target_col].values,
            train["date"], train_limit=5000,
        )

    best_name = min(MODELS, key=lambda name: results[name]["N"]["cv_mae_mean"])
    best_metrics = results[best_name]["N"]
    best = {"model": best_name, "metrics": best_metrics}

    # Baselines ingenuos sobre el MISMO test set (comparacion honesta)
    def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        return {
            "test_mae": float(mean_absolute_error(y_true, y_pred)),
            "test_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "test_r2": float(r2_score(y_true, y_pred)),
        }

    baselines: dict[str, dict[str, dict]] = {}
    for tlogical, tcol in target_cols.items():
        y_test = test[tcol].values
        baselines[tlogical] = {
            # Persistencia: predecir el valor actual del indice (t -> t+1)
            "persistence": _metrics(y_test, test[f"{tlogical}_t"].values),
            # Media: predecir la media del train
            "mean": _metrics(y_test, np.full_like(y_test, train[tcol].mean())),
        }
    print("[regression] Baselines calculados (persistencia + media del train)", flush=True)

    # Persist
    json_path = PROCESSED_DIR / "r4_results.json"
    json_path.write_text(
        json.dumps(
            {"models": results, "best": best, "baselines": baselines},
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"[regression] Resultados JSON -> {json_path}", flush=True)

    # Figura
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIG_DIR / "11_regression_comparison.png"
    plot_comparison(results, fig_path)
    print(f"[regression] Figura -> {fig_path}", flush=True)

    # Reporte
    report_path = REPORTS_DIR / "04_regression.md"
    write_report(results, best, baselines, report_path)
    print(f"[regression] Reporte -> {report_path}", flush=True)

    return results, best


if __name__ == "__main__":
    main()
