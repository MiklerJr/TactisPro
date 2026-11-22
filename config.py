import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = Path(os.environ.get("DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

RAW_MATCHES = RAW_DIR / "charting-m-matches.csv"
RAW_POINTS_2010s = RAW_DIR / "charting-m-points-2010s.csv"
RAW_POINTS_2020s = RAW_DIR / "charting-m-points-2020s.csv"

CLEAN_DATASET = PROCESSED_DIR / "dataset_clean.parquet"
MOMENTUM_DATASET = PROCESSED_DIR / "momentum_series.parquet"

MATCHES_DATASET = PROCESSED_DIR / "matches.parquet"
GAMES_DATASET = PROCESSED_DIR / "games.parquet"
GAME_FEATURES_DATASET = PROCESSED_DIR / "game_features.parquet"
GAME_FEATURES_RESOLVED_DATASET = PROCESSED_DIR / "game_features_resolved.parquet"

RANDOM_SEED = 42

GAME_WINDOW_SIZES = [2, 3, 5]