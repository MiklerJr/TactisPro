"""R2: compara el aporte de las ventanas para predecir el ganador del juego siguiente."""

import json

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import GAME_FEATURES_RESOLVED_DATASET, GAMES_DATASET, PROCESSED_DIR, REPORTS_DIR


def evaluate() -> dict:
    games = pd.read_parquet(GAMES_DATASET, columns=[
        "match_id", "game_num", "date", "will_serve_next", "p1_wins_next_game",
    ]).dropna(subset=["date", "p1_wins_next_game"])
    games["year"] = pd.to_datetime(games["date"]).dt.year
    features = pd.read_parquet(GAME_FEATURES_RESOLVED_DATASET)
    dataset = features.merge(games, on=["match_id", "game_num"], validate="many_to_one")
    results = {}
    base = ["will_serve_next"]
    for window, rows in dataset.groupby("window_size"):
        train = rows[rows["year"] < 2023]
        test = rows[rows["year"] >= 2023]
        target = "p1_wins_next_game"
        train_y = train[target].astype(int)
        test_y = test[target].astype(int)
        dynamic = [column for column in features if column.endswith(f"_W{window}")]
        comparison = {}
        for name, predictors in [("Saque previsto", base), ("Saque previsto y variables dinámicas", base + dynamic)]:
            model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression())
            model.fit(train[predictors], train_y)
            prob = model.predict_proba(test[predictors])[:, 1]
            comparison[name] = {
                "roc_auc": float(roc_auc_score(test_y, prob)),
                "brier": float(brier_score_loss(test_y, prob)),
                "accuracy": float(accuracy_score(test_y, prob >= 0.5)),
            }
        results[str(window)] = {"train": len(train), "test": len(test), "metrics": comparison}
    return results


def main() -> None:
    results = evaluate()
    (PROCESSED_DIR / "r2_predictive_contribution.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# R2 · Contribución predictiva de las variables dinámicas", "",
        "Objetivo: predecir si el jugador A gana el siguiente juego. Se compara la misma "
        "regresión logística usando el saque previsto y agregando las 16 variables dinámicas "
        "de cada ventana. La imputación y el escalado se ajustan en entrenamiento.", "",
        "Entrenamiento: partidos anteriores a 2023. Prueba: partidos desde 2023. Dentro de "
        "cada ventana, ambos modelos se evalúan sobre las mismas observaciones.", "",
        "| W | Entrenamiento | Prueba | Entradas | AUC-ROC | Brier | Exactitud |",
        "|---:|---:|---:|---|---:|---:|---:|",
    ]
    for window, result in results.items():
        for model, metrics in result["metrics"].items():
            lines.append(
                f"| {window} | {result['train']:,} | {result['test']:,} | {model} | "
                f"{metrics['roc_auc']:.4f} | {metrics['brier']:.4f} | {metrics['accuracy']:.4f} |"
            )
    lines.extend(["", "El contraste mide el aporte conjunto de las variables para este objetivo y modelo."])
    (REPORTS_DIR / "02_contribucion_predictiva.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
