# R2: Dataset de Games (Hard Court)

## 1. Resumen general

| Metrica | Valor |
|---|---|
| Superficie | Hard Court |
| Periodo | 2010-01-08 a 2026-04-06 |
| Partidos | 3,571 |
| Games | 91,753 |
| Feature rows | 275,259 (3 ventanas x game) |
| Win rate (server) | 80.9% |
| Ventanas | W=2, W=3, W=5 (per-player per-role) |
| Target | p1_wins_next_game (96.1% coverage) |
| Tablas | matches, games, game_features |
| Features por ventana | 54 antes de resolución; 48 después de eliminar 6 features redundantes |

---

## 2. Features de ventana (Per-Player Per-Role)

Las features de ventana se separan por jugador (p1/p2) y rol (serve/return). El dataset bruto contiene 18 features por ventana; después de eliminar las 2 features de win rate redundantes por ventana quedan 16 features por ventana en `game_features_resolved.parquet`.

### Formulas de features (per-player)

Para un game `i` en el match, con ventana `W`:
- `window = games[i-W*2:i]` (los ultimos `W*2` games del match, sin incluir el game actual)

| Feature | Formula | Tipo |
|---|---|---|
| p1_serve_wr_W | `mean(won_game donde current_server == p1_id)` | Serve |
| p1_return_wr_W | `mean(1 - won_game donde current_returner == p1_id)` | Return |
| p1_aces_W | `mean(aces donde current_server == p1_id)` | Serve |
| p1_df_W | `mean(double_faults donde current_server == p1_id)` | Serve |
| p1_1st_pct_W | `mean(first_serve_in_pct donde current_server == p1_id)` | Serve |
| p1_svr_pts_won_W | `mean(svr_pts_won_pct donde current_server == p1_id)` | Serve |
| p1_rally_avg_W | `mean(rally_avg donde p1 sirve o recibe)` | Per-game |
| p1_winners_W | `mean(winners donde p1 sirve o recibe)` | Per-game |
| p1_ue_W | `mean(unforced_errors donde p1 sirve o recibe)` | Per-game |
| p1_fe_W | `mean(forced_errors donde p1 sirve o recibe)` | Per-game |
| p2_* | Mismas formulas con `p2_id` | - |

### Features de contexto (match, 3 por ventana)

| Feature | Formula |
|---|---|
| rally_avg_W | `mean(rally_avg de todos los games en window)` |
| error_rate_W | `sum(unforced_errors + forced_errors) / sum(n_points)` |
| win_trend_W | `mean(won_game de primera mitad) - mean(won_game de segunda mitad)` |

### Ventanas

| Ventana | Games lookback | Descripcion |
|---|---|---|
| W=2 | 4 games | Ultimos 2 saques/returns por jugador |
| W=3 | 6 games | Ultimos 3 saques/returns por jugador |
| W=5 | 10 games | Ultimos 5 saques/returns por jugador |

### Verificacion de leakage

Test automatico en `src/features/sliding_window.py::test_no_leakage()`:

- **Verifica** que para un game N, la ventana solo incluye games `1..N-1`
- **Verifica** que el ultimo game en la ventana es `N-1` (no `N`)
- **Verifica** que la feature manual calculada con la ventana NO usa el game actual
- **Resultado**: PASSED (verificado con match sample)

```
window = indicators.iloc[start:test_idx]  # start = test_idx - W*2, test_idx = current game index
# window.index[-1] == test_idx - 1  (ultimo game en ventana)
# test_idx NOT in window.index      (game actual excluido)
```

---

## 6. Correlaciones (Con Rolling Stats)

### Features de ventana vs p1_wins_next_game (W=3)

#### Top 10 features

| Feature | Correlacion | p-value | N validos | Tipo |
|---|---|---|---|---|
| p2_svr_pts_won_W3 | **-0.066** | 1.14e-78 | 81,040 | Rolling stats |
| p1_return_wr_W3 | **+0.063** | 5.91e-73 | 81,040 | Win rate |
| p2_serve_wr_W3 | **-0.063** | 5.91e-73 | 81,040 | Win rate |
| p1_svr_pts_won_W3 | **+0.063** | 8.51e-76 | 84,611 | Rolling stats |
| p1_serve_wr_W3 | **+0.062** | 2.30e-72 | 84,611 | Win rate |
| p2_return_wr_W3 | **-0.062** | 2.30e-72 | 84,611 | Win rate |
| p2_aces_W3 | **-0.027** | 8.98e-15 | 81,040 | Rolling stats |
| p1_aces_W3 | **+0.026** | 1.49e-14 | 84,611 | Rolling stats |
| p1_df_W3 | **-0.017** | 5.25e-07 | 84,611 | Rolling stats |
| p2_df_W3 | **+0.016** | 6.09e-06 | 81,040 | Rolling stats |

#### Interpretacion

- **svr_pts_won_pct** (rolling) es la feature con mayor correlacion (-0.066), superando a los win rates
- **Aces** muestran correlacion significativa (+0.026/-0.027)
- **DFs** tambien muestran correlacion significativa (-0.017/+0.016)
- **1st serve %** tiene correlacion moderada (+0.012/-0.013)
- **Winners y errores** tienen correlaciones debiles pero significativas
- Las correlaciones positivas de p1_* y negativas de p2_* son esperadas porque el target es "p1 wins next game"

---

## 7. Modelo de BD

### Estructura de 3 tablas

```
matches (3,571 rows)              <-- metadata del partido + player stats (p1/p2)
  |
  |-- games (91,753 rows)          <-- 1 row por game, con target + player stats
  |     |
  |     |-- game_features (275,259 rows)  <-- 3 rows por game (W=2,3,5)
  |
```

| Tabla | Rows | Columnas clave |
|---|---|---|
| `matches.parquet` | 3,571 | match_id, player_1, player_2, surface, tournament, date, best_of, p1_player_id, p2_player_id, p1_rank, p2_rank, p1_age, p2_age, p1_elo, p2_elo, p1_hand, p2_hand |
| `games.parquet` | 91,753 | match_id, game_num, won_game, p1_wins_next_game, p1_player_id, p2_player_id, p1_rank, p2_rank, p1_age, p2_age, p1_elo, p2_elo, current_server, current_returner, rank_diff, elo_diff |
| `game_features.parquet` | 275,259 | match_id, game_num, window_size, p1_serve_wr_W, p1_return_wr_W, p2_serve_wr_W, p2_return_wr_W, rally_avg_W, error_rate_W, win_trend_W |

### Features de jugadores

| Feature | Fuente | Descripcion |
|---|---|---|
| p1_player_id | Derivado | ID numerico unico por jugador 1 |
| p2_player_id | Derivado | ID numerico unico por jugador 2 |
| p1_rank | Sackmann | Ranking ATP del jugador 1 al momento del partido |
| p2_rank | Sackmann | Ranking ATP del jugador 2 al momento del partido |
| p1_age | Sackmann | Edad del jugador 1 al momento del partido |
| p2_age | Sackmann | Edad del jugador 2 al momento del partido |
| rank_diff | Derivado | p1_rank - p2_rank |
| age_diff | Derivado | p1_age - p2_age |
| p1_elo | Tennis Abstract | yElo del jugador 1 |
| p2_elo | Tennis Abstract | yElo del jugador 2 |
| elo_diff | Derivado | p1_elo - p2_elo |

### Features de servidor actual (dinamico)

| Feature | Descripcion |
|---|---|
| current_server | Nombre del jugador que sirve en este game |
| current_returner | Nombre del jugador que recibe en este game |
| current_server_hand | Mano del servidor actual |
| current_returner_hand | Mano del receptor actual |

---

## 8. Notas metodologicas

### Leakage removido

Las siguientes columnas fueron removidas del dataset de entrenamiento porque contienen informacion que no esta disponible al momento de la prediccion:

- `svr_pts_won_pct`, `aces`, `double_faults`, `first_serve_in_pct`
- `rally_avg`, `n_points`, `winners`, `forced_errors`, `unforced_errors`, `net_approaches`

### Target definido

- **p1_wins_next_game**: Binario (0/1) que indica si el Player 1 gana el proximo game
- **Coverage**: 96.1% (solo el ultimo game de cada match no tiene target)
- **Razon**: Los roles de sacador/receptor se intercambian entre games, por lo que el target por jugador es mas estable que el target por rol

### Cold-start handling

- `game_num` se usa para identificar los primeros games donde las ventanas no tienen suficiente historial

### Fuentes de datos

1. **Sackmann ATP archive** (`Aneeshers/tennis-sackmann-archive`): Datos historicos de ranking y edad por partido.
2. **Tennis Abstract yElo**: Ratings ELO actuales por jugador (actualizado semanalmente).
3. **Nuestro dataset**: Generado a partir de `dataset_clean.parquet` (point-level).

### Criterios de inclusion

- Solo partidos en superficie **Hard Court**
- Periodo: 2010-2024
- Solo partidos con datos completos de puntos

### Limitaciones

- Player stats (rank, age) dependen del matching por nombre (puede haber discrepancias)
- ELO es actual (2026), no historico por partido
- Cobertura de rank/age: ~83% (algunos partidos no tienen datos en Sackmann)
- Cobertura de ELO: ~98% (con estimacion rank→ELO para jugadores retirados)
- Features de contexto (rally_avg, errors) no se pueden separar por jugador sin volver a datos de puntos
- **Correlaciones weak**: Las features de ventana per-player per-role muestran correlaciones modestas con el target (|r| < 0.08), lo cual es esperado en datos de tenis ruidosos y refleja que el resultado de un game individual depende de muchos factores instante a instante

### Proximo paso

Entrenar modelo de prediccion (Fase 3) usando `game_features` como input y `p1_wins_next_game` como target.
