"""Parámetros compartidos por el entrenamiento y la inferencia de R6."""

from config import PROCESSED_DIR

HISTORY = 5
INPUTS = ["M_A", "M_B", "N"]
MODEL_PATH = PROCESSED_DIR / "r6_gru.pt"
