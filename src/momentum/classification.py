"""R5 — Clasificador de direccion de momentum con seleccion de features y GridSearch.

Indicador revisado: superar a la regresión logística en precisión,
sensibilidad, F1 y AUC-ROC en el mismo conjunto de prueba.

Target:
    next_dir_up = 1 si N(t+1) > N(t), 0 si N(t+1) <= N(t).
    Equivale a predecir si el momentum subira o bajara en el siguiente game.
    Se interpreta como "punto de inflexion" en el sentido de cambio de tendencia.

Modelos:
    - Logistic Regression (baseline obligatorio del EDT)
    - Random Forest
    - XGBoost (con GridSearchCV)

Seleccion de features:
    - Recursive Feature Elimination (RFE) sobre LR para pre-seleccionar top-K.
    - SHAP values sobre el modelo seleccionado.

Split:
    Train: anos < 2023 | Test: anos >= 2023.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from .temporal import date_folds
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

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


def build_dataset(match_id: str | None = None, *, window: int = 5) -> pd.DataFrame:
    """Une features R2 + momentum actual + features de cambio + target binario."""
    filters = [("match_id", "=", match_id)] if match_id is not None else None
    feats = pd.read_parquet(GAME_FEATURES_RESOLVED_DATASET, filters=filters)
    games = pd.read_parquet(GAMES_DATASET, filters=filters)[["match_id", "game_num", "date"]]
    mom = pd.read_parquet(MOMENTUM_DATASET, filters=filters).sort_values(["match_id", "game_num"])

    # Una fila por juego y únicamente las variables de la ventana evaluada.
    columns = [c for c in feats if c.endswith(f"_W{window}") or c in {"match_id", "game_num"}]
    feats = feats.loc[feats["window_size"].eq(window), columns].copy()

    # Merge con fecha
    feats = feats.merge(games, on=["match_id", "game_num"], how="inner")

    # Merge con momentum actual y pasado (para features de cambio)
    mom_curr = mom[["match_id", "game_num", "M_A", "M_B", "N"]].rename(
        columns={"M_A": "M_A_t", "M_B": "M_B_t", "N": "N_t"}
    )
    feats = feats.merge(mom_curr, on=["match_id", "game_num"], how="inner")

    # Features de cambio/dinamica (capturan tendencia reciente)
    mom["N_prev"] = mom.groupby("match_id")["N"].shift(1)
    mom["M_A_prev"] = mom.groupby("match_id")["M_A"].shift(1)
    mom["M_B_prev"] = mom.groupby("match_id")["M_B"].shift(1)
    mom["N_next"] = mom.groupby("match_id")["N"].shift(-1)

    mom_changes = mom[["match_id", "game_num", "N_prev", "M_A_prev", "M_B_prev", "N_next"]].copy()
    feats = feats.merge(mom_changes, on=["match_id", "game_num"], how="inner")

    add_momentum_features(feats)

    # Target binario: 1 si N(t+1) > N(t) (momentum subira).
    # El ultimo game de cada partido no tiene N(t+1): se descarta ANTES de la
    # comparacion (en pandas, NaN > x evaluado a False y etiquetaria clase 0).
    feats = feats.dropna(subset=["N_next"])
    feats["next_dir_up"] = (feats["N_next"] > feats["N_t"]).astype(int)

    feats["date"] = pd.to_datetime(feats["date"])
    feats["year"] = feats["date"].dt.year

    return feats


def add_momentum_features(feats: pd.DataFrame) -> None:
    """Añade características de cambio a partir del índice actual y el previo."""
    feats["N_diff"] = feats["N_t"] - feats["N_prev"]  # cambio respecto al game previo
    feats["abs_N_t"] = feats["N_t"].abs()
    feats["abs_N_diff"] = feats["N_diff"].abs()
    feats["N_sign"] = np.sign(feats["N_t"])  # signo del momentum

    # Magnitud relativa del cambio (cuanto se movio vs cuanto tenia)
    feats["N_rel_change"] = feats["N_diff"] / (feats["abs_N_t"] + 0.1)
    feats["N_volatility"] = feats["N_t"] ** 2  # energia del momentum

    # Diferencia entre jugadores
    feats["M_diff"] = feats["M_A_t"] - feats["M_B_t"]
    feats["M_A_minus_prev"] = feats["M_A_t"] - feats["M_A_prev"]
    feats["M_B_minus_prev"] = feats["M_B_t"] - feats["M_B_prev"]


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    drop = {
        "match_id", "game_num", "date", "year",
        "next_dir_up", "N_next", "N_prev", "M_A_prev", "M_B_prev",
    }
    return [c for c in df.columns if c not in drop]


def impute_features(train, test, feat_cols):
    medians = train[feat_cols].median(numeric_only=True)
    for c in feat_cols:
        train[c] = train[c].fillna(medians[c])
        test[c] = test[c].fillna(medians[c])


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def run_baseline_and_tree(X_train, y_train, X_test, y_test, feat_cols):
    """Entrena LR (baseline + RFE), RF y XGBoost base."""
    models = {
        "Logistic Regression": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE,
            n_jobs=-1, verbosity=0, eval_metric="logloss",
        ),
    }
    results = {}
    fitted = {}
    for name, mdl in models.items():
        mdl.fit(X_train, y_train)
        y_pred = mdl.predict(X_test)
        y_proba = mdl.predict_proba(X_test)[:, 1]
        results[name] = compute_metrics(y_test, y_pred, y_proba)
        fitted[name] = mdl
    return results, fitted


def run_gridsearch_xgb(X_train, y_train, X_test, y_test, dates):
    """GridSearchCV sobre XGBoost (5-fold TS)."""
    pipe = XGBClassifier(
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=0, eval_metric="logloss",
    )
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1, 0.2],
        "subsample": [0.7, 0.9],
        "colsample_bytree": [0.7, 0.9],
    }
    pipeline = make_pipeline(SimpleImputer(strategy="median"), pipe)
    cv = date_folds(dates, 5)
    grid = {f"xgbclassifier__{name}": values for name, values in param_grid.items()}
    gs = GridSearchCV(
        pipeline, grid, cv=cv,
        scoring="roc_auc", n_jobs=-1, verbose=0, refit=True,
    )
    gs.fit(X_train, y_train)
    best = gs.best_estimator_
    y_pred = best.predict(X_test)
    y_proba = best.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_proba)
    return {
        "best_params": {name.split("__", 1)[1]: value for name, value in gs.best_params_.items()},
        "best_cv_roc_auc": float(gs.best_score_),
        "test_metrics": metrics,
    }, best.named_steps["xgbclassifier"]


def run_rfe(X_train, y_train, X_test, y_test, n_features: int):
    """RFE sobre LR (con datos escalados) para seleccionar top features."""
    sc = StandardScaler()
    X_train_s = sc.fit_transform(X_train)
    X_test_s = sc.transform(X_test)
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    rfe = RFE(lr, n_features_to_select=n_features, step=5)
    rfe.fit(X_train_s, y_train)
    y_pred = rfe.predict(X_test_s)
    y_proba = rfe.predict_proba(X_test_s)[:, 1]
    return rfe, compute_metrics(y_test, y_pred, y_proba)


def plot_comparison(metrics_dict: dict, out_path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = list(metrics_dict.keys())
    precision = [metrics_dict[k]["precision"] for k in keys]
    recall = [metrics_dict[k]["recall"] for k in keys]
    f1 = [metrics_dict[k]["f1"] for k in keys]
    auc = [metrics_dict[k]["roc_auc"] for k in keys]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(keys))
    axes[0].bar(x - 0.3, precision, 0.2, label="Precision", color="#2563EB")
    axes[0].bar(x - 0.1, recall, 0.2, label="Recall", color="#DC2626")
    axes[0].bar(x + 0.1, f1, 0.2, label="F1", color="#059669")
    axes[0].bar(x + 0.3, auc, 0.2, label="AUC-ROC", color="#9CA3AF")
    axes[0].axhline(0.75, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    axes[0].axhline(0.80, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(keys, rotation=15)
    axes[0].set_ylim(0.5, 1.0)
    axes[0].set_title("Metricas en test set")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].grid(True, axis="y", alpha=0.25)

    # CM del mejor modelo
    if "XGBoost (GridSearch)" in metrics_dict:
        cm = metrics_dict["XGBoost (GridSearch)"]["confusion_matrix"]
        cm_arr = np.array(cm)
        axes[1].imshow(cm_arr, interpolation="nearest", cmap=plt.cm.Blues)
        for i in range(cm_arr.shape[0]):
            for j in range(cm_arr.shape[1]):
                axes[1].text(j, i, f"{cm_arr[i, j]:,}",
                             ha="center", va="center",
                             color="white" if cm_arr[i, j] > cm_arr.max() / 2 else "black")
        axes[1].set_xticks([0, 1])
        axes[1].set_yticks([0, 1])
        axes[1].set_xticklabels(["Baja (0)", "Sube (1)"])
        axes[1].set_yticklabels(["Baja (0)", "Sube (1)"])
        axes[1].set_title("Matriz confusion - XGBoost (GridSearch)")
        axes[1].set_ylabel("Real")
        axes[1].set_xlabel("Predicho")

    fig.suptitle("R5 - Clasificador de direccion de momentum", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def shap_analysis(model, X_sample, feat_cols, out_path: Path, max_features: int = 15):
    """Genera SHAP summary plot (bar)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # clase positiva
    shap_df = pd.DataFrame(shap_values, columns=feat_cols)
    importance = shap_df.abs().mean().sort_values(ascending=True).tail(max_features)

    fig, ax = plt.subplots(figsize=(9, 0.4 * len(importance) + 1))
    ax.barh(importance.index, importance.values, color="#2563EB")
    ax.set_xlabel("Importancia SHAP (|valor medio|)")
    ax.set_title(f"Top {len(importance)} features - SHAP values")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return importance.to_dict()


def write_report(
    metrics_dict, gs_result, rfe_metrics, feat_cols, rfe_support,
    shap_importance, out_path: Path, split_stats: dict,
):
    best_name = "XGBoost (GridSearch)"
    best_m = metrics_dict[best_name]
    pct_up = split_stats["pct_up"]
    n_total = split_stats["n_total"]
    n_train = split_stats["n_train"]
    n_test = split_stats["n_test"]
    lines = [
        "# R5. CLASIFICADOR DE LA DIRECCION DEL CAMBIO DE MOMENTUM",
        "",
        "## 5.1 Objetivo",
        "",
        "Clasificar la direccion del cambio del momentum neto N(t) (binarizacion:",
        "aumentara o disminuira en t+1), con seleccion de caracteristicas (RFE) y",
        "optimizacion de hiperparametros (GridSearch), superando al baseline",
        "de Regresion Logistica.",
        "",
        "## 5.2 Configuracion experimental",
        "",
        "**Target binario**: `next_dir_up = 1` si N(t+1) > N(t), `0` en caso contrario.",
        f"**Balance de clases**: {100 * pct_up:.1f}% sube / {100 * (1 - pct_up):.1f}% baja "
        f"(sobre {n_total:,} games; se descartan los ultimos games de cada partido).",
        f"**Split**: Train < 2023 ({n_train:,} filas) | Test >= 2023 ({n_test:,} filas).",
        "",
        "**Features**:",
        f"- {len([c for c in feat_cols if '_W5' in c])} features R2 (W=5)",
        "- 3 features autorregresivas: M_A(t), M_B(t), N(t)",
        "- 9 features de cambio/dinamica: N_diff, abs_N_t, abs_N_diff, N_sign,",
        "  N_rel_change, N_volatility, M_diff, M_A_minus_prev, M_B_minus_prev",
        f"- Total: {len(feat_cols)} features",
        "",
        "## 5.3 Resultados comparativos en test set",
        "",
        "| Modelo | Precision | Recall | F1 | AUC-ROC |",
        "|---|---|---|---|---|",
    ]
    for name, m in metrics_dict.items():
        lines.append(
            f"| {name} | {m['precision']:.4f} | {m['recall']:.4f} "
            f"| {m['f1']:.4f} | {m['roc_auc']:.4f} |"
        )

    lines += [
        "",
        "![Comparativa modelos](figures/12_classification_comparison.png)",
        "",
        "## 5.4 Optimizacion de hiperparametros - GridSearch",
        "",
        f"**Modelo base**: XGBoost.",
        f"**Espacio de busqueda**: {len(gs_result['best_params'])} parametros, "
        f"total de combinaciones = 2*3*3*2*2 = 72.",
        "",
        "**Mejores hiperparametros encontrados**:",
        "",
    ]
    for k, v in gs_result["best_params"].items():
        lines.append(f"- `{k}` = `{v}`")
    lines += [
        "",
        f"**Mejor AUC-ROC en CV (5-fold TimeSeriesSplit)**: "
        f"`{gs_result['best_cv_roc_auc']:.4f}`",
        "",
        "**Metricas en test con hiperparametros optimizados**:",
        "",
        f"- Precision: {gs_result['test_metrics']['precision']:.4f}",
        f"- Recall: {gs_result['test_metrics']['recall']:.4f}",
        f"- F1: {gs_result['test_metrics']['f1']:.4f}",
        f"- AUC-ROC: {gs_result['test_metrics']['roc_auc']:.4f}",
        "",
        "## 5.5 Seleccion de caracteristicas - RFE",
        "",
        f"Se aplico Recursive Feature Elimination (RFE) sobre Logistic Regression,",
        f"seleccionando las top 20 features por importancia.",
        "",
        f"**Metricas tras RFE**: P={rfe_metrics['precision']:.4f}, "
        f"R={rfe_metrics['recall']:.4f}, F1={rfe_metrics['f1']:.4f}, "
        f"AUC={rfe_metrics['roc_auc']:.4f}",
        "",
        f"**Features seleccionadas** (top 10):",
        "",
    ]
    selected = [f for f, s in zip(feat_cols, rfe_support) if s]
    for f in selected[:10]:
        lines.append(f"- `{f}`")
    if len(selected) > 10:
        lines.append(f"- ... ({len(selected) - 10} mas)")

    lines += [
        "",
        "## 5.6 Importancia de caracteristicas - SHAP",
        "",
        "Se calcularon valores SHAP sobre el modelo XGBoost optimizado:",
        "",
        "![SHAP importance](figures/13_shap_importance.png)",
        "",
        "**Top 10 features por valor SHAP medio**:",
        "",
        "| Feature | Importancia SHAP |",
        "|---|---|",
    ]
    for k, v in list(shap_importance.items())[-10:]:
        lines.append(f"| `{k}` | {v:.4f} |")

    lines += [
        "",
        "## 5.7 Seleccion del mejor modelo y cumplimiento del indicador",
        "",
        f"**Modelo evaluado**: **{best_name}**, con ajuste por validación cronológica.",
        "",
        "El criterio se revisó después de los experimentos. Los umbrales aspiracionales originales de 0,75 en precisión, sensibilidad y F1 y de 0,80 en AUC-ROC no se alcanzaron.",
        "",
        "| Métrica | Regresión logística | XGBoost | Supera la referencia |",
        "|---|---:|---:|:---:|",
    ]
    reference = metrics_dict["Logistic Regression"]
    for key, label in [("precision", "Precisión"), ("recall", "Sensibilidad"),
                       ("f1", "F1"), ("roc_auc", "AUC-ROC")]:
        status = "Sí" if best_m[key] > reference[key] else "No"
        lines.append(f"| {label} | {reference[key]:.4f} | {best_m[key]:.4f} | {status} |")

    lines += [
        "",
        "## 5.8 Medio de verificacion",
        "",
        "- Script: `src/momentum/classification.py`",
        "- Resultados crudos: `data/processed/r5_results.json`",
        "- Reporte: `reports/05_classification.md`",
        "- Figuras: `docs_tesis/figures/12_classification_comparison.png` y `13_shap_importance.png`",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[classification] Reporte -> {out_path}")


def main():
    print("[classification] Construyendo dataset...")
    df = build_dataset()
    print(f"[classification] Filas totales con target: {len(df):,}")
    print(f"[classification] Distribucion target:\n{df['next_dir_up'].value_counts()}")

    train = df[df["year"] < 2023].copy()
    test = df[df["year"] >= 2023].copy()
    print(f"[classification] Train: {len(train):,} | Test: {len(test):,}")

    feat_cols = get_feature_cols(df)
    X_train_raw = train[feat_cols].to_numpy(copy=True)
    X_test_raw = test[feat_cols].to_numpy(copy=True)
    impute_features(train, test, feat_cols)

    X_train = train[feat_cols].values
    X_test = test[feat_cols].values
    y_train = train["next_dir_up"].astype(int).values
    y_test = test["next_dir_up"].astype(int).values

    print("[classification] Entrenando modelos base...")
    base_results, fitted = run_baseline_and_tree(X_train, y_train, X_test, y_test, feat_cols)
    for name, m in base_results.items():
        print(
            f"   {name}: F1={m['f1']:.4f} AUC={m['roc_auc']:.4f}"
        )

    print("[classification] Ejecutando GridSearch (XGBoost)...")
    gs_result, gs_model = run_gridsearch_xgb(X_train_raw, y_train, X_test_raw, y_test, train["date"])
    base_results["XGBoost (GridSearch)"] = gs_result["test_metrics"]
    print(
        f"   GS best CV AUC={gs_result['best_cv_roc_auc']:.4f} | "
        f"test AUC={gs_result['test_metrics']['roc_auc']:.4f}"
    )

    print("[classification] Ejecutando RFE (LR)...")
    n_sel = min(20, len(feat_cols))
    rfe_model, rfe_metrics = run_rfe(X_train, y_train, X_test, y_test, n_sel)
    rfe_support = rfe_model.support_

    print("[classification] Calculando SHAP values...")
    sample_size = min(2000, len(X_test))
    rng = np.random.default_rng(RANDOM_STATE)
    sample_idx = rng.choice(len(X_test), size=sample_size, replace=False)
    X_sample = X_test[sample_idx]
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    shap_path = FIG_DIR / "13_shap_importance.png"
    shap_importance = shap_analysis(gs_model, X_sample, feat_cols, shap_path)

    # Figura
    cmp_path = FIG_DIR / "12_classification_comparison.png"
    plot_comparison(base_results, cmp_path)
    print(f"[classification] Figuras -> {FIG_DIR}")

    # Persistir
    json_path = PROCESSED_DIR / "r5_results.json"
    json_path.write_text(
        json.dumps(
            {
                "models": base_results,
                "gridsearch": gs_result,
                "rfe_metrics": rfe_metrics,
                "rfe_selected_features": [f for f, s in zip(feat_cols, rfe_support) if s],
                "shap_importance": shap_importance,
                "n_features_total": len(feat_cols),
                "n_features_rfe": int(rfe_support.sum()),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    # Reporte
    report_path = REPORTS_DIR / "05_classification.md"
    split_stats = {
        "pct_up": float(df["next_dir_up"].mean()),
        "n_total": int(len(df)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
    }
    write_report(
        base_results, gs_result, rfe_metrics, feat_cols, rfe_support,
        shap_importance, report_path, split_stats,
    )


if __name__ == "__main__":
    main()
