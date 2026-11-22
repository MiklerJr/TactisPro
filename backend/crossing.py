"""Inferencia CPU de la GRU de R6 con los pesos exportados de PyTorch."""

import numpy as np
from scipy.special import expit

from src.momentum.crossing_model import MODEL_PATH


def load_weights():
    if not MODEL_PATH.is_file():
        return None

    import torch

    state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    return {name: tensor.numpy() for name, tensor in state.items()}


def forecast_crossing(windows: np.ndarray, weights: dict) -> np.ndarray:
    """Aplica las compuertas de PyTorch en orden reset, update y new."""
    recurrent = weights["gru.weight_hh_l0"]
    hidden = np.zeros((len(windows), recurrent.shape[1]), dtype=np.float32)
    for inputs in windows.transpose(1, 0, 2):
        input_gates = inputs @ weights["gru.weight_ih_l0"].T + weights["gru.bias_ih_l0"]
        hidden_gates = hidden @ recurrent.T + weights["gru.bias_hh_l0"]
        input_reset, input_update, input_new = np.split(input_gates, 3, axis=1)
        hidden_reset, hidden_update, hidden_new = np.split(hidden_gates, 3, axis=1)
        reset = expit(input_reset + hidden_reset)
        update = expit(input_update + hidden_update)
        candidate = np.tanh(input_new + reset * hidden_new)
        hidden = (1 - update) * candidate + update * hidden

    linear = hidden @ weights["output.weight"].T + weights["output.bias"]
    return np.logaddexp(0, linear[:, 0]) + 1
