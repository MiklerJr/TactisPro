import unittest

import numpy as np
import torch

from backend.crossing import forecast_crossing
from src.momentum.crossing_forecast import CrossingGRU
from src.momentum.crossing_model import HISTORY, INPUTS


class CrossingInferenceTest(unittest.TestCase):
    def test_numpy_matches_pytorch_for_batches_and_single_games(self):
        model = CrossingGRU().eval()
        windows = np.random.default_rng(5).normal(size=(32, HISTORY, len(INPUTS))).astype(np.float32)
        weights = {name: tensor.numpy() for name, tensor in model.state_dict().items()}
        with torch.inference_mode():
            expected = model(torch.from_numpy(windows)).numpy()

        np.testing.assert_allclose(forecast_crossing(windows, weights), expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(forecast_crossing(windows[:1], weights), expected[:1], rtol=1e-6, atol=1e-6)
