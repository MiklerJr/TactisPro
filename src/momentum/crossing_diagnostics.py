"""R6 exploratory diagnostics restricted to training and validation through 2022."""

import json

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, roc_auc_score
from threadpoolctl import threadpool_limits

from config import GAMES_DATASET, MOMENTUM_DATASET, PROCESSED_DIR, RANDOM_SEED
from .crossing_forecast import HISTORY, INPUTS, MODEL_PATH, CrossingGRU
from .crossings import momentum_crossings

CONTEXT = ["set_num", "set1", "set2", "gm1", "gm2", "best_of", "will_serve_next", "p1_wins_game"]
SIGNALS = ["distance", "toward_zero", "sign_age", "game_num", *CONTEXT]


def examples() -> tuple[pd.DataFrame, np.ndarray]:
    games = pd.read_parquet(GAMES_DATASET, columns=["match_id", "game_num", "date", *CONTEXT])
    games["year"] = pd.to_datetime(games["date"]).dt.year
    momentum = pd.read_parquet(MOMENTUM_DATASET, columns=["match_id", "game_num", *INPUTS])
    data = momentum.merge(games[games["year"].le(2022)], on=["match_id", "game_num"], validate="one_to_one")
    data = data.sort_values(["match_id", "game_num"])
    rows, histories = [], []
    for _, match in data.groupby("match_id", sort=False):
        match = match.set_index("game_num", drop=False)
        crosses = momentum_crossings(match["N"]).index.to_numpy()
        values = match[INPUTS].to_numpy(dtype=np.float32)
        first_nonzero = match["N"].ne(0).idxmax() if match["N"].ne(0).any() else np.inf
        for end, row in enumerate(match.itertuples(index=False)):
            if not HISTORY - 1 <= end < len(match) - 1:
                continue
            game = row.game_num
            position = crosses.searchsorted(game, side="right")
            previous = crosses[position - 1] if position else first_nonzero
            rows.append({
                **{name: getattr(row, name) for name in CONTEXT},
                "game_num": game,
                "year": row.year,
                "distance": abs(row.N),
                "toward_zero": abs(values[end - 1, -1]) - abs(row.N),
                "sign_age": game - previous if previous <= game else np.nan,
                "target": crosses[position] - game if position < len(crosses) else np.nan,
            })
            histories.append(values[end - HISTORY + 1:end + 1])
    return pd.DataFrame(rows), np.asarray(histories)


def grouped_reference(train, validation):
    """Training medians by distance, direction, sign duration, serve and set score."""
    def bins(data):
        return pd.DataFrame({
            "distance": pd.cut(data["distance"], [-np.inf, .1, .3, .6, np.inf], labels=False),
            "toward": data["toward_zero"].gt(0).astype(int),
            "age": pd.cut(data["sign_age"], [-np.inf, 2, 5, np.inf], labels=False).fillna(-1),
            "server": data["will_serve_next"].fillna(-1),
            "set_score": np.sign(data["gm1"] - data["gm2"]).fillna(0),
        })

    trained = bins(train).assign(target=train["target"])
    groups = bins(validation)
    predictions = pd.Series(np.nan, index=validation.index)
    for count in (5, 3, 2, 1):
        columns = list(groups.columns[:count])
        aggregate = trained.groupby(columns)["target"].agg(["median", "size"]).reset_index()
        supported = aggregate.loc[aggregate["size"].ge(30), [*columns, "median"]]
        matched = groups.merge(supported, on=columns, how="left", validate="many_to_one")["median"]
        predictions = predictions.fillna(pd.Series(matched.to_numpy(), index=groups.index))
    return predictions.fillna(train["target"].median()).to_numpy()


def diagnostics(frame, error):
    bins = {
        "horizonte observado": ("target", [0, 2, 5, 10, np.inf]),
        "distancia a cero": ("distance", [-np.inf, .1, .3, .6, np.inf]),
        "juegos con el signo vigente": ("sign_age", [-np.inf, 2, 5, np.inf]),
    }
    clean = frame.assign(error=error)
    summaries = {}
    for label, (column, cuts) in bins.items():
        groups = clean.groupby(pd.cut(clean[column], cuts), observed=True)["error"]
        summaries[label] = [
            {"interval": str(interval), "n": int(len(values)), "mae": float(values.mean()),
             "error_share": float(values.sum() / clean["error"].sum())}
            for interval, values in groups
        ]
    return summaries


def evaluate():
    data, histories = examples()
    train = data[data["year"].lt(2022)]
    valid = data[data["year"].eq(2022)]
    train_event, valid_event = train.dropna(subset=["target"]), valid.dropna(subset=["target"])
    gru = CrossingGRU()
    gru.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    gru.eval()
    with torch.inference_mode():
        baseline = gru(torch.from_numpy(histories[valid_event.index])).numpy()
    error = np.abs(valid_event["target"].to_numpy() - baseline)
    horizons = {
        "Mediana constante": mean_absolute_error(valid_event["target"], np.full(len(valid_event), train_event["target"].median())),
        "GRU publicada": float(error.mean()),
        "Mediana condicionada": mean_absolute_error(valid_event["target"], grouped_reference(train_event, valid_event)),
    }
    for depth in (2, 4, 6):
        reg = HistGradientBoostingRegressor(
            loss="absolute_error", max_depth=depth, max_iter=120, early_stopping=False,
            random_state=RANDOM_SEED,
        )
        reg.fit(train_event[SIGNALS], train_event["target"])
        horizons[f"Árboles con contexto, profundidad {depth}"] = mean_absolute_error(
            valid_event["target"], reg.predict(valid_event[SIGNALS])
        )
    train_occurred = train["target"].notna()
    valid_occurred = valid["target"].notna()
    clf = HistGradientBoostingClassifier(max_depth=4, max_iter=120, early_stopping=False, random_state=RANDOM_SEED)
    clf.fit(train[SIGNALS], train_occurred)
    probabilities = {
        "Frecuencia de entrenamiento": np.full(len(valid), train_occurred.mean()),
        "Árboles con contexto": clf.predict_proba(valid[SIGNALS])[:, 1],
    }
    return {
        "train_eligible": len(train), "train_with_crossing": len(train_event),
        "validation_eligible": len(valid), "validation_with_crossing": len(valid_event),
        "validation_horizon_mae": horizons,
        "gru_error_diagnostics": diagnostics(valid_event, error),
        "validation_probabilities": {
            name: {"brier": brier_score_loss(valid_occurred, p), "log_loss": log_loss(valid_occurred, p),
                   "roc_auc": roc_auc_score(valid_occurred, p)}
            for name, p in probabilities.items()
        },
    }


def main():
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        results = evaluate()
    output = json.dumps(results, ensure_ascii=False, indent=2) + "\n"
    (PROCESSED_DIR / "r6_diagnostics.json").write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
