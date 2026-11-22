"""Cliente HTTP del backend de predicción."""

import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import urlopen

import streamlit as st

BACKEND_URL = os.environ.get("TACTISPRO_BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")


def _get(path: str) -> dict:
    try:
        with urlopen(f"{BACKEND_URL}{path}", timeout=20) as response:
            return json.load(response)
    except HTTPError as exc:
        raise RuntimeError(f"El backend respondió con HTTP {exc.code}.") from exc
    except (URLError, TimeoutError) as exc:
        raise RuntimeError(f"No se pudo conectar con el backend en {BACKEND_URL}.") from exc


@st.cache_data(ttl=60, show_spinner=False)
def fetch_metrics() -> dict:
    return _get("/metrics")


def fetch_game(match_id: str, game_num: int) -> dict:
    return _get(f"/matches/{quote(match_id, safe='')}/games/{game_num}")
