"""Limpieza, parseo y construcción del dataset estructurado punto a punto.

Carga los CSVs de puntos de Sackmann, aplica el parser de códigos, une la
metadata de partidos y deriva el estado del marcador por punto. Guarda el
resultado en parquet y produce el reporte de calidad de datos (R1).
"""

from __future__ import annotations

import pandas as pd

from config import RAW_MATCHES, RAW_POINTS_2010s, RAW_POINTS_2020s, CLEAN_DATASET, PROCESSED_DIR
from .parser import apply_parser_to_frame


def load_points() -> pd.DataFrame:
    # Carga y concatena los puntos de las décadas 2010 y 2020
    d10 = pd.read_csv(RAW_POINTS_2010s, low_memory=False)
    d20 = pd.read_csv(RAW_POINTS_2020s, low_memory=False)

    #concatenamos
    df = pd.concat([d10, d20], ignore_index=True)
    # llenamos con vacios las filas que luego vamos a parsear
    for col in ("1st", "2nd", "Notes"):
        df[col] = df[col].fillna("").astype(str)
    return df

# las superficies validas del dataset
VALID_SURFACES = {"Hard", "Clay", "Grass", "Carpet"}


def load_matches() -> pd.DataFrame:
    m = pd.read_csv(RAW_MATCHES, low_memory=False)
    # eliminamos duplicados y nos quedamos con el primero -> reseteamos los indices
    m = m.drop_duplicates(subset=["match_id"], keep="first").reset_index(drop=True)
    # validamos los valores dentro de "Surface" sino completamos con "Unknown"
    m["Surface"] = m["Surface"].where(m["Surface"].str.strip().isin(VALID_SURFACES), "Unknown")
    m["Best of"] = pd.to_numeric(m["Best of"], errors="coerce").fillna(3.0)
    return m


def parse_points(df: pd.DataFrame) -> pd.DataFrame:
    """Expande 1st/2nd en variables estructuradas."""
    parsed = apply_parser_to_frame(df)
    flat = pd.json_normalize(parsed.tolist())
    return pd.concat([df, flat], axis=1)


def _pts_to_idx(value: str):
    """Convierte un componente de marcador en índice de puntos."""
    if value in ("0", "1", "2", "3", "4"):
        return int(value)
    return {"0": 0, "15": 1, "30": 2, "40": 3, "AD": 4, "A": 4}.get(value)


def derive_score(df: pd.DataFrame) -> pd.DataFrame:
    """Deriva el estado del juego a partir de la columna Pts."""
    parts = df["Pts"].astype(str).str.split("-", expand=True)
    if parts.shape[1] < 2:
        parts[1] = ""
    df["pts_srv_raw"] = parts[0]
    df["pts_ret_raw"] = parts[1]
    df["is_tiebreak"] = df["pts_srv_raw"].astype(str).str.fullmatch(r"\d+").astype(bool) & \
        df["pts_ret_raw"].astype(str).str.fullmatch(r"\d+").astype(bool) & \
        ~(df["pts_srv_raw"].astype(str).isin(["0", "15", "30", "40"]) &
          df["pts_ret_raw"].astype(str).isin(["0", "15", "30", "40"]))
    df["pt_srv"] = df["pts_srv_raw"].apply(_pts_to_idx)
    df["pt_ret"] = df["pts_ret_raw"].apply(_pts_to_idx)
    non_tb = ~df["is_tiebreak"]
    df["deuce"] = (df["pt_srv"] >= 3) & (df["pt_ret"] >= 3) & (df["pt_srv"] == df["pt_ret"]) & non_tb
    df["ad_srv"] = (df["pt_srv"] >= 4) & (df["pt_srv"] > df["pt_ret"]) & non_tb
    df["ad_ret"] = (df["pt_ret"] >= 4) & (df["pt_ret"] > df["pt_srv"]) & non_tb
    # Punto de quiebre: el restador está a un punto de ganar el juego.
    break_pt = (df["pt_ret"] == 3) & (df["pt_srv"].le(2)) | (df["ad_ret"])
    df["is_break_point"] = break_pt & ~df["is_tiebreak"]
    return df


def merge_matches(df: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Une metadata y hace la atribución de jugador según quien sirve."""
    df = df.merge(
        matches,
        on="match_id",
        how="left",
        suffixes=("", "_match"),
    )
    df["Date"] = pd.to_datetime(df["Date"].astype(str).str[:8], errors="coerce")
    one_serving = df["Svr"].astype(int) == 1
    df["current_server"] = df["Player 1"].where(one_serving, df["Player 2"])
    df["current_returner"] = df["Player 2"].where(one_serving, df["Player 1"])
    df["current_server_hand"] = df["Pl 1 hand"].where(one_serving, df["Pl 2 hand"])
    df["current_returner_hand"] = df["Pl 2 hand"].where(one_serving, df["Pl 1 hand"])
    df["isSvrWinner"] = (df["PtWinner"].astype(int) == df["Svr"].astype(int)).astype(int)
    df["isRtrWinner"] = 1 - df["isSvrWinner"]
    return df


def end_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Completa finalizadores perdidos desde PtWinner y marca incidencias."""
    df["ending_inferred"] = 0
    missing_end = df["ending"].isna() & df["parse_issue"].isna() & ~df["isDouble"]
    df["ending_inferred"] = missing_end.astype(int)
    df.loc[missing_end & df["isRtrWinner"].eq(1), "parse_issue"] = "no_ending_ret_won"
    df.loc[missing_end & df["isSvrWinner"].eq(1), "parse_issue"] = "no_ending_svr_won"
    df["parse_issue"] = df["parse_issue"].fillna("ok")
    return df


def run_pipeline() -> pd.DataFrame:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print("[1/5] Cargando puntos ...")
    df = load_points()
    print(f"  {len(df):,} puntos")
    print("[2/5] Parseando códigos ...")
    df = parse_points(df)
    print("[3/5] Derivando marcador ...")
    df = derive_score(df)
    print("[4/5] Uniendo metadata de partidos ...")
    matches = load_matches()
    df = merge_matches(df, matches)
    df = end_flags(df)
    n_raw = len(df)
    df = df.drop_duplicates(subset=["match_id", "Pt"], keep="first")
    n_dup = n_raw - len(df)
    (PROCESSED_DIR / "metadata.json").write_text(
        __import__("json").dumps(
            {"raw_points": n_raw, "duplicates_removed": int(n_dup),
             "points_clean": int(len(df))}, indent=2),
        encoding="utf-8")
    print("[5/5] Guardando dataset limpio ...")
    df.to_parquet(CLEAN_DATASET, index=False)
    print(f"  Guardado en {CLEAN_DATASET} ({len(df):,} filas, "
          f"{n_dup:,} duplicados eliminados)")
    return df


if __name__ == "__main__":
    df = run_pipeline()
    print(df.shape)
