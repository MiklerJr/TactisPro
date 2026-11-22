"""R10: contrasta las alertas del replay con cruces retrospectivos."""

import json

import pandas as pd

from config import CLEAN_DATASET, GAMES_DATASET, MOMENTUM_DATASET, PROCESSED_DIR, REPORTS_DIR
from dashboard.replay import resolve_live_state
from .crossings import momentum_crossings


def reference_crossings(net: pd.Series) -> dict:
    """Genera la referencia retrospectiva mediante pares sucesivos de valores no nulos."""
    nonzero = net.dropna()
    nonzero = nonzero[nonzero.ne(0)]
    return {
        int(game): 1 if value > 0 else -1
        for (_, previous), (game, value) in zip(nonzero.items(), list(nonzero.items())[1:])
        if (previous < 0 < value) or (previous > 0 > value)
    }


def evaluate() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    dates = pd.read_parquet(GAMES_DATASET, columns=["match_id", "date"]).drop_duplicates("match_id")
    test_ids = set(dates.loc[pd.to_datetime(dates["date"]).dt.year.ge(2023), "match_id"])
    momentum = pd.read_parquet(
        MOMENTUM_DATASET, columns=["match_id", "game_num", "N"],
    ).sort_values(["match_id", "game_num"])
    momentum = momentum[momentum["match_id"].isin(test_ids)]
    points = pd.read_parquet(CLEAN_DATASET, columns=["match_id", "Gm#", "Pt"])
    points = points[points["match_id"].isin(test_ids)].sort_values(["match_id", "Gm#", "Pt"])
    point_groups = points.groupby("match_id", sort=False)
    events = []
    closures = []
    missing = 0
    for match_id, match in momentum.groupby("match_id", sort=False):
        net = match.set_index("game_num")["N"]
        reference = reference_crossings(net)
        if match_id not in point_groups.indices:
            missing += 1
            continue
        match_points = point_groups.get_group(match_id).reset_index(drop=True)
        ends = match_points["Gm#"].ne(match_points["Gm#"].shift(-1)).to_numpy().nonzero()[0] + 1
        actual = {}
        for last_point in ends:
            state = resolve_live_state(match_points, int(last_point))
            if not state["game_closed"]:
                continue
            game_num = state["completed"]
            revealed = net.loc[:game_num]
            crossing = momentum_crossings(revealed).get(game_num)
            closures.append((match_id, game_num, int(last_point), crossing))
            if crossing is not None:
                actual[game_num] = int(crossing)
        for game in sorted(reference.keys() | actual.keys()):
            events.append({
                "match_id": match_id, "game_num": game, "observed_direction": reference.get(game),
                "replay_direction": actual.get(game),
            })

    event_log = pd.DataFrame(
        events, columns=["match_id", "game_num", "observed_direction", "replay_direction"]
    )
    closure_log = pd.DataFrame(
        closures, columns=["match_id", "game_num", "points_revealed", "alert_direction"]
    )
    closure_log["alert_direction"] = closure_log["alert_direction"].astype("Int64")
    expected = event_log["observed_direction"].notna()
    emitted = event_log["replay_direction"].notna()
    matches = expected & emitted & event_log["observed_direction"].eq(event_log["replay_direction"])
    tp, fp, fn = int(matches.sum()), int((emitted & ~matches).sum()), int((expected & ~matches).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    metrics = {
        "test_start_year": 2023, "matches": momentum["match_id"].nunique() - missing,
        "matches_without_points": missing, "closed_games": len(closure_log), "reference_events": int(expected.sum()),
        "emitted_alerts": int(emitted.sum()), "true_positives": tp, "false_positives": fp,
        "false_negatives": fn, "precision": precision, "recall": recall,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }
    return metrics, event_log, closure_log


def main() -> None:
    metrics, events, closures = evaluate()
    events.to_csv(PROCESSED_DIR / "r10_alert_events.csv", index=False)
    closures.to_csv(PROCESSED_DIR / "r8_replay_closures.csv", index=False)
    (PROCESSED_DIR / "r10_alert_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = [
        "# R10 · Evaluación de las alertas de cruce por cero", "",
        "Se reprodujeron partidos históricos del conjunto de prueba (2023 en adelante). "
        "El evaluador recorre los cierres de juego del registro punto a punto, aplica "
        "el mismo cálculo que utiliza el dashboard sobre el prefijo revelado "
        "y registra cada cruce una vez.", "",
        "La referencia se calcula retrospectivamente sobre la serie registrada, comparando "
        "pares consecutivos de valores distintos de cero. La coincidencia exige el mismo "
        "partido, juego y dirección. Una dirección equivocada cuenta como falso positivo "
        "para la alerta emitida y falso negativo para el cruce observado.", "",
        f"Partidos evaluados: **{metrics['matches']:,}**. Juegos cerrados: **{metrics['closed_games']:,}**.", "",
        "| Cruces observados | Alertas emitidas | VP | FP | FN | Precisión | Sensibilidad | F1 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {metrics['reference_events']:,} | {metrics['emitted_alerts']:,} | "
        f"{metrics['true_positives']:,} | {metrics['false_positives']:,} | {metrics['false_negatives']:,} | "
        f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} |",
        "",
        "El experimento mide la concordancia de un detector de cruces observados. "
        "El error del pronóstico de futuros cruces se evalúa por separado en R6.", "",
        "Registro por evento: `data/processed/r10_alert_events.csv`.",
        "Registro por cierre en el evaluador del replay: `data/processed/r8_replay_closures.csv`.",
    ]
    (REPORTS_DIR / "06_evaluacion_alertas.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
