"""Preserve the index construction while randomizing game order within role."""

import json

import numpy as np
import pandas as pd
from scipy.stats import chi2

from config import CLEAN_DATASET, GAMES_DATASET, MOMENTUM_DATASET, PROCESSED_DIR, RANDOM_SEED
from src.features.game_aggregation import aggregate_games
from src.features.sliding_window import _build_indicators
from .engine import _compute_player_momentum

WINDOW = 5
MATCHES = 200
PERMUTATIONS = 99
INDICATORS = [
    "p1_serve_won", "p1_return_won", "p1_svr_pts_won_pct", "p1_first_serve_in_pct", "p1_double_faults",
    "p2_svr_pts_won_pct", "p2_first_serve_in_pct", "p2_double_faults",
]


def reconstruct(indicators):
    # Roll only the inputs consumed by the published EWM index.
    windows = np.full(indicators.shape, np.nan)
    for column, values in enumerate(indicators.T):
        for end in range(1, len(values)):
            section = values[max(0, end - 2 * WINDOW):end]
            present = section[~np.isnan(section)]
            if present.size:
                windows[end, column] = present.mean()
    past = pd.DataFrame(windows)
    a = list(range(5))
    past[8] = 1 - past[1]
    past[9] = 1 - past[0]
    b = [8, 9, 5, 6, 7]
    return (
        _compute_player_momentum(past, a[:-1], a[-1:])["score"].to_numpy()
        - _compute_player_momentum(past, b[:-1], b[-1:])["score"].to_numpy()
    )


def statistics(net):
    centered = net - net.mean()
    variance = centered @ centered
    if variance == 0:
        return 0.0, False
    n = len(net)
    lags = np.arange(1, max(5, min(10, n // 4)) + 1)
    correlations = np.array([centered[k:] @ centered[:-k] / variance for k in lags])
    q = n * (n + 2) * np.sum(correlations ** 2 / (n - lags))
    return float(correlations[0]), bool(chi2.sf(q, len(lags)) < .05)


def evaluate():
    rng = np.random.default_rng(RANDOM_SEED)
    net = pd.read_parquet(MOMENTUM_DATASET, columns=["match_id", "game_num", "N"])
    dates = pd.read_parquet(GAMES_DATASET, columns=["match_id", "date"]).drop_duplicates("match_id")
    test_matches = dates.loc[pd.to_datetime(dates["date"]).dt.year.ge(2023), "match_id"]
    sizes = net[net["match_id"].isin(test_matches)].groupby("match_id").size()
    ids = sorted(rng.choice(sizes[sizes.ge(20)].index, size=MATCHES, replace=False))
    games = aggregate_games(pd.read_parquet(CLEAN_DATASET, filters=[("match_id", "in", ids)]))
    expected = net[net["match_id"].isin(ids)].set_index(["match_id", "game_num"])["N"]
    observed = []
    reference = []
    max_difference = 0.
    for match_id, match in games.groupby("match_id", sort=True):
        match = match.sort_values("game_num")
        first = match.iloc[0]
        inputs = _build_indicators(match, first["current_server"], first["current_returner"])[INDICATORS].to_numpy()
        rebuilt = reconstruct(inputs)
        original = expected.loc[match_id].reindex(match["game_num"]).to_numpy()
        np.testing.assert_allclose(rebuilt, original, rtol=1e-9, atol=1e-9, err_msg=match_id)
        max_difference = max(max_difference, float(np.max(np.abs(rebuilt - original))))
        observed.append(statistics(original))
        strata = match.groupby(["current_server", "is_tiebreak"], dropna=False).indices.values()
        replicas = []
        for _ in range(PERMUTATIONS):
            order = np.arange(len(match))
            for positions in strata:
                order[positions] = rng.permutation(positions)
            replicas.append(statistics(reconstruct(inputs[order])))
        reference.append(replicas)
    observed = np.asarray(observed)
    reference = np.asarray(reference)
    reference_means = reference.mean(axis=0)
    names = ["lag1", "ljung_box_rejection_rate"]
    results = {}
    for column, name in enumerate(names):
        actual = observed[:, column].mean()
        null = reference_means[:, column]
        results[name] = {
            "observed": float(actual), "null_mean": float(null.mean()),
            "null_interval95": np.percentile(null, [2.5, 97.5]).tolist(),
            "observed_minus_null": float(actual - null.mean()),
            "one_sided_permutation_p": float((1 + np.count_nonzero(null >= actual)) / (len(null) + 1)),
        }
    return {
        "matches": len(observed), "eligible_matches": len(sizes[sizes.ge(20)]), "draws": PERMUTATIONS,
        "seed": RANDOM_SEED, "match_ids": ids, "max_reconstruction_difference": max_difference, "results": results,
    }


if __name__ == "__main__":
    result = evaluate()
    output = json.dumps(result, indent=2) + "\n"
    (PROCESSED_DIR / "r3_permutation_audit.json").write_text(output, encoding="utf-8")
    print(result["results"])
