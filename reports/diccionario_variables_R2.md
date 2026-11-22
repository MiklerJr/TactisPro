# Diccionario de Variables - R2

**Resultado**: R2 - Conjunto de caracteristicas dinamicas derivadas
**Fecha**: 2026-10-01

Este documento describe las **60 variables** utilizadas como entrada
de los modelos R4 y R5. Se divide en:

- **48 features dinamicas** (ventanas W=2/3/5) en `game_features_resolved.parquet`
- **12 variables no-ventana** (player stats + serve state) en `games.parquet`

---

## 1. Features Dinamicas (48)

Cada feature existe en 3 ventanas: **W=2, W=3, W=5** (4-10 games lookback).
El sufijo `_Wx` indica la ventana.

| Grupo | Variable (base) | W disponibles | Descripcion | Unidad | Fuente |
|---|---|---|---|---|---|
| Win Rates (post-MC) | `p1_serve_wr` | W=2, 3, 5 | Win rate de P1 cuando sirve (ventana W) | ratio 0-1 | games.parquet (won_game, current_server) |
| Context | `rally_avg` | W=2, 3, 5 | Promedio de golpes por punto en ventana | shots/point | games.parquet (rallyCount) |
| Win Rates (post-MC) | `p1_return_wr` | W=2, 3, 5 | Win rate de P1 cuando resta (ventana W) | ratio 0-1 | games.parquet (1-won_game, current_returner) |
| Serve Quality | `p2_1st_pct` | W=2, 3, 5 | % primer servicio dentro de P2 | ratio 0-1 | games.parquet (1stIn, current_server) |
| Rally Quality (post-MC) | `fe` | W=2, 3, 5 | Promedio de errores forzados por game en ventana | FE/game | games.parquet (isForced) |
| Serve Quality | `p2_svr_pts_won` | W=2, 3, 5 | % puntos ganados al saque por P2 | ratio 0-1 | games.parquet (svr_pts_won_pct, current_server) |
| Context | `error_rate` | W=2, 3, 5 | Tasa de errores (UE+FE) por punto en ventana | ratio 0-1 | games.parquet (unforced_errors, forced_errors, n_points) |
| Rally Quality (post-MC) | `winners` | W=2, 3, 5 | Promedio de winners (rally winners) por game en ventana | winners/game | games.parquet (isRallyWinner) |
| Serve Quality | `p1_df` | W=2, 3, 5 | Promedio de dobles faltas por game de servicio de P1 | DF/game | games.parquet (isDouble, current_server) |
| Serve Quality | `p2_df` | W=2, 3, 5 | Promedio de dobles faltas por game de servicio de P2 | DF/game | games.parquet (isDouble, current_server) |
| Serve Quality | `p1_1st_pct` | W=2, 3, 5 | % primer servicio dentro de P1 | ratio 0-1 | games.parquet (1stIn, current_server) |
| Serve Quality | `p1_aces` | W=2, 3, 5 | Promedio de aces por game de servicio de P1 | aces/game | games.parquet (isAce, current_server) |
| Serve Quality | `p2_aces` | W=2, 3, 5 | Promedio de aces por game de servicio de P2 | aces/game | games.parquet (isAce, current_server) |
| Rally Quality (post-MC) | `ue` | W=2, 3, 5 | Promedio de errores no forzados por game en ventana | UE/game | games.parquet (isUnforced) |
| Context | `win_trend` | W=2, 3, 5 | Tendencia de victorias: win rate mitad reciente - mitad anterior | ratio -1 a +1 | games.parquet (won_game) |
| Serve Quality | `p1_svr_pts_won` | W=2, 3, 5 | % puntos ganados al saque por P1 | ratio 0-1 | games.parquet (svr_pts_won_pct, current_server) |

### 1.1 Cobertura por ventana

| Ventana | Games lookback | # features por ventana | Total (x3) |
|---|---|---|---|
| W=2 | 4 games | 16 | 48 |
| W=3 | 6 games | 16 | 48 |
| W=5 | 10 games | 16 | 48 |

**Nota**: el total 48 features dinamicas cubre las 3 ventanas. Por ventana
se emiten 16 features: 2 win rates + 8 serve quality (4 metricas x p1/p2) + 3
rally quality (winners/ue/fe) + 3 context (rally_avg/error_rate/win_trend).

---

## 2. Player Stats (9)

Atributos estaticos del jugador al momento del partido.

| Variable | Descripcion | Unidad | Fuente |
|---|---|---|---|
| `p1_rank` | Ranking ATP de P1 al momento del partido | entero (1-N) | Sackmann (LOCF) |
| `p2_rank` | Ranking ATP de P2 al momento del partido | entero (1-N) | Sackmann (LOCF) |
| `p1_age` | Edad de P1 al momento del partido | años (float) | Sackmann (LOCF) |
| `p2_age` | Edad de P2 al momento del partido | años (float) | Sackmann (LOCF) |
| `p1_elo` | Rating ELO de P1 | float (~1500) | Tennis Abstract yElo |
| `p2_elo` | Rating ELO de P2 | float (~1500) | Tennis Abstract yElo |
| `rank_diff` | p1_rank - p2_rank (positivo = P1 mejor rankeado) | entero | derivado |
| `age_diff` | p1_age - p2_age (positivo = P1 mayor) | años | derivado |
| `elo_diff` | p1_elo - p2_elo (positivo = P1 mejor ELO) | float | derivado |

**Cobertura** (verificada en `src/features/coverage_improvement.py`):

| Campo | Cobertura | Estrategia de imputacion |
|---|---|---|
| p1_rank | 96.3% | Sackmann + LOCF (max 3 anos) |
| p2_rank | 96.6% | Sackmann + LOCF |
| p1_age | 96.3% | Sackmann + LOCF |
| p2_age | 96.6% | Sackmann + LOCF |
| p1_elo | 98.1% | Tennis Abstract yElo |
| p2_elo | 97.7% | Tennis Abstract yElo |
| p1_hand | 100.0% | Datos originales |
| p2_hand | 100.0% | Datos originales |

---

## 3. Serve State (3 dinamicas + 2 hand)

Estado del servicio por game.

| Variable | Descripcion | Unidad | Fuente |
|---|---|---|---|
| `current_server` | ID del jugador que saca en el game actual | string | raw match data |
| `current_returner` | ID del jugador que resta en el game actual | string | raw match data |
| `current_server_hand` | Mano de saque del servidor (L=zurdo, R=diestro) | char | raw match data |
| `current_returner_hand` | Mano del restador | char | raw match data |
| `will_serve_next` | Indicador de quién sacará en el siguiente game (1=p1 sacará) | 0/1 | derivado de game_num |

---

## 4. Metadata (12)

Columnas de contexto e identificacion.

| Variable | Descripcion | Unidad | Fuente |
|---|---|---|---|
| `match_id` | Identificador unico del partido | string | raw match data |
| `game_num` | Numero de game dentro del partido (1, 2, 3, ...) | entero >=1 | raw match data |
| `window_size` | Tamano de ventana W (2, 3 o 5) | entero | config.py |
| `set_num` | Numero de set en el que ocurre el game | entero >=1 | raw match data |
| `is_tiebreak` | True si el game es tiebreak | bool | raw match data |
| `surface` | Superficie de la cancha (Hard, Clay, Grass, Unknown) | string | raw match data |
| `tournament` | Nombre del torneo | string | raw match data |
| `date` | Fecha del partido | YYYY-MM-DD | raw match data |
| `best_of` | Sets para ganar el partido (3 o 5) | entero | raw match data |
| `won_game` | 1 si el sacador gano el game, 0 si lo perdio | 0/1 | agregado |
| `p1_wins_game` | 1 si P1 gano el game, 0 si lo perdio | 0/1 | derivado de won_game |
| `p1_wins_next_game` | 1 si P1 gana el siguiente game (target) | 0/1 (NaN en ultimo game) | shift(-1) de p1_wins_game |

---

## 5. Resumen total

| Categoria | Cantidad | Archivo |
|---|---|---|
| Features dinamicas | 48 | `data/processed/game_features_resolved.parquet` |
| Player stats | 9 | `data/processed/games.parquet` |
| Serve state | 5 (3 dinamicos + 2 hand) | `data/processed/games.parquet` |
| Metadata | 12 | `data/processed/games.parquet` |
| **TOTAL** | **74** | (de las cuales 57 son input para modelos) |

**Para los modelos R4/R5 se usan 48 features dinamicas + 3 autorregresivas
(M_A(t), M_B(t), N(t)) + 12 player stats/serve state = 51+ features de entrada.**

---

## 6. Reglas de derivacion

Todas las features dinamicas se calculan con `src/features/sliding_window.py`
usando la funcion `sliding_windows()`. Reglas clave:

1. **Sin leakage**: la ventana para game N incluye solo games 1..N-1 (no N).
2. **Cold-start**: en los primeros games del partido la ventana es parcial
   (usando `min_periods=1`), generando NaN que se imputan despues con la
   mediana del train.
3. **Per-game aggregate**: `winners`, `ue`, `fe`, `rally_avg`, `error_rate`,
   `win_trend` se emiten UNA vez por ventana (sin prefijo p1_/p2_), porque
   el dato fuente (`isRallyWinner`, `isUnforced`, etc.) no lleva atribucion
   al jugador (ver seccion 2.4 de `r2_caracteristicas_dinamicas.md`).
4. **Per-role service stats**: `p1_*` y `p2_*` SI se preservan para aces, df,
   1st_pct, svr_pts_won porque el filtro `is_pX_serve` es True solo en la
   mitad de los games (servicio alternado).

## 7. Reproducibilidad

```bash
python -m scripts.build_game_dataset    # genera game_features.parquet
python -m src.features.multicollinearity  # genera game_features_resolved.parquet
```

**Tests de integridad disponibles**:

- `src/features/sliding_window.py::test_no_leakage()` - PASSED
- `src/features/multicollinearity.py::detect_multicollinearity()` - max |r| = 0.0 post-resolution
- `src/features/significance_test.py::test_feature_significance()` - 35/48 con p<0.05
- `src/features/train_test_validation.py::validate_train_test()` - 20 features estables