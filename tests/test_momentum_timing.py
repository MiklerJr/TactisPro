import unittest

import numpy as np
import pandas as pd

from src.momentum.crossings import momentum_crossings
from src.momentum.engine import _compute_player_momentum
from src.momentum.temporal import date_folds


class MomentumTimingTests(unittest.TestCase):
    def test_future_games_do_not_change_published_scores(self):
        games = pd.DataFrame({
            "positive": [np.nan, 0.2, 0.4, 0.3, 0.9],
            "negative": [0.0, 0.0, 0.2, 0.0, 10.0],
        })
        full = _compute_player_momentum(games, ["positive"], ["negative"])
        for count in range(1, len(games)):
            prefix = _compute_player_momentum(games.iloc[:count], ["positive"], ["negative"])
            pd.testing.assert_frame_equal(prefix, full.iloc[:count])
        self.assertTrue(full["score"].between(0, 1).all())

    def test_crossing_is_reported_when_the_new_sign_is_observed(self):
        scores = pd.Series([0.0, -0.5, 0.0, -0.4, 0.0, 0.2, -0.1], index=range(1, 8))
        self.assertEqual(momentum_crossings(scores).to_dict(), {6: 1.0, 7: -1.0})
        self.assertTrue(momentum_crossings(scores.loc[:5]).empty)

    def test_folds_use_complete_dates_in_chronological_order(self):
        dates = np.array(["2022-01-04", "2022-01-01", "2022-01-02", "2022-01-02",
                          "2022-01-03", "2022-01-04"], dtype="datetime64[D]")
        for train, validation in date_folds(dates, 2):
            self.assertLess(dates[train].max(), dates[validation].min())
            np.testing.assert_array_equal(validation, np.flatnonzero(np.isin(dates, dates[validation])))


if __name__ == "__main__":
    unittest.main()
