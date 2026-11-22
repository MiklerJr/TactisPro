# Documentación Requerida - P1 y P2 (EDT Tesis)

> **Objetivo**: Este documento lista exhaustivamente todo lo que debe incluirse
> en el documento formal de resultados de la tesis para los puntos 1 y 2 del EDT:
>
> - **P1**: Dataset limpio, normalizado y estructurado punto a punto
> - **P2**: Conjunto de características dinámicas derivadas mediante ingeniería de variables

---

## PUNTO 1: Dataset limpio, normalizado y estructurado punto a punto

### 1.1 Descripción general del dataset

| Campo | Valor a documentar |
|---|---|
| Fuente original | Match Charting Project de Jeff Sackmann |
| Archivos fuente | `charting-m-points-2010s.csv`, `charting-m-points-2020s.csv` |
| Periodo temporal del dataset utilizado | 2010-01-08 a 2026-04-06 |
| Total de puntos crudos | 904,707 |
| Duplicados removidos | 579 |
| Puntos limpios finales | 904,128 |
| Total de partidos | 5,569 |
| Total de juegos (Hard Court) | 91,753 |
| Total de filas en features (3 ventanas × game) | 275,259 |

### 1.2 Distribución por superficie

| Superficie | Puntos | Porcentaje |
|---|---|---|
| Hard | 569,959 | 63.0% |
| Clay | 232,162 | 25.7% |
| Grass | 101,626 | 11.2% |
| Unknown | 381 | 0.04% |

**Justificación del filtro Hard Court**: Solo se usa Hard Court porque...

### 1.3 Pipeline de limpieza (parser)

- **Porcentaje parseado sin incidencias**: 99.6%
- **Incidencias del parser**:
  - `ok`: 900,941 (99.6%)
  - `special_point`: 2,553 (0.3%)
  - `stopped_challenge`: 325 (0.0%)
  - `no_ending_ret_won`: 250 (0.0%)
  - `no_ending_svr_won`: 40 (0.0%)
  - `unresolved_double_fault`: 19 (0.0%)

### 1.4 Indicadores derivados clave

| Indicador | Valor |
|---|---|
| % puntos ganados por el sacador | 64.26% |
| % aces | 8.34% |
| % dobles faltas | 3.41% |
| % errores no forzados | 32.05% |
| % errores forzados | 30.17% |
| % winners de rally | 22.98% |
| % puntos de quiebre | 8.06% |
| Golpes promedio por punto | 4.10 |
| % primer servicio dentro | 62.79% |

### 1.5 Tests de integridad (objetivamente verificables)

**Resultado: 7/7 tests PASS**

| Test | Resultado | Detalle |
|---|---|---|
| Columnas críticas sin nulos | PASS | 18/18 columnas con 0% nulos |
| `game_num` contiguo por match | PASS | Todos los matches tienen game_num secuencial |
| `won_game` consistente | PASS | 0.69% mismatch (636/91,753) < 1% tolerancia |
| `Date` parseable | PASS | 99.98% fechas válidas |
| Superficie válida | PASS | Solo {Hard, Clay, Grass, Unknown} |
| `isAce` y `isDouble` excluyentes | PASS | 0 puntos con ambos flags |
| `rallyCount` no negativo | PASS | 0 valores negativos |

### 1.6 Nulos e inconsistencias

- **Celdas nulas totales**: 2,943,476 (4.79%)
- **Columnas con nulos**: 18 (todas opcionales, ninguna crítica)

### 1.7 Columnas críticas (0% nulos requerido)

1. `match_id` - Identificador único del partido
2. `Pt` - Número de punto dentro del partido
3. `Gm#` - Número de juego del partido
4. `Set1`, `Set2` - Sets ganados por cada jugador
5. `Player 1`, `Player 2` - Nombres de jugadores
6. `Surface` - Superficie
7. `current_server`, `current_returner` - Servidor/receptor del punto
8. `isSvrWinner` - Ganador del punto
9. `isAce`, `isDouble` - Indicadores de saque
10. `isRallyWinner`, `isUnforced`, `isForced` - Finalizadores
11. `rallyCount` - Número de golpes
12. `net_approaches` - Subidas a la red

### 1.8 Columnas opcionales (aceptan nulos)

Lista completa en `reports/data_dictionary.md`

### 1.9 Cobertura de player stats (después de LOCF)

| Campo | Cobertura | Estrategia |
|---|---|---|
| p1_rank | 96.3% | Sackmann + LOCF (max 3 años) |
| p2_rank | 96.6% | Sackmann + LOCF |
| p1_age | 96.3% | Sackmann + LOCF |
| p2_age | 96.6% | Sackmann + LOCF |
| p1_elo | 98.1% | Tennis Abstract yElo |
| p2_elo | 97.7% | Tennis Abstract yElo |
| p1_hand | 100.0% | Datos originales |
| p2_hand | 100.0% | Datos originales |

### 1.10 Target definido

- **Nombre**: `p1_wins_next_game`
- **Tipo**: Binario (0/1)
- **Coverage**: 96.1% (último game de cada match sin target)
- **Cálculo**: `shift(-1)` de `p1_wins_game` dentro de cada match
- **Justificación**: Target por jugador (no por rol de sacador) porque los roles se alternan

---

## PUNTO 2: Conjunto de características dinámicas derivadas

### 2.1 Resumen de features

| Categoría | Cantidad |
|---|---|
| Features por ventana (antes) | 18 |
| Features redundantes eliminadas | 2 por ventana (6 en total) |
| Features útiles (después) | 48 |
| Ventanas | W=2, W=3, W=5 |

### 2.2 Tipos de features

#### Per-player features (P1)
- `p1_serve_wr_W` - Win rate cuando sirve
- `p1_return_wr_W` - Win rate cuando recibe
- `p1_aces_W` - Aces promedio por servicio
- `p1_df_W` - Dobles faltas promedio
- `p1_1st_pct_W` - % primer servicio
- `p1_svr_pts_won_W` - % puntos ganados al saque
- `p1_winners_W` - Winners promedio
- `p1_ue_W` - Errores no forzados promedio
- `p1_fe_W` - Errores forzados promedio

#### Per-player features (P2, sin redundancia)
- `p2_aces_W` - Aces promedio (P2)
- `p2_df_W` - Dobles faltas (P2)
- `p2_1st_pct_W` - % primer servicio (P2)
- `p2_svr_pts_won_W` - % puntos ganados al saque (P2)
- `p2_winners_W` - Winners (P2)
- `p2_ue_W` - Errores no forzados (P2)
- `p2_fe_W` - Errores forzados (P2)

#### Context features
- `error_rate_W` - Tasa de errores por punto
- `win_trend_W` - Tendencia de wins

### 2.3 Fórmulas exactas

Para un game `i` en el match, con ventana `W`:
- `window = games[i-W*2:i]` (últimos `W*2` games, sin incluir game actual)

| Feature | Fórmula |
|---|---|
| p1_serve_wr_W | `mean(won_game donde current_server == p1_id)` |
| p1_return_wr_W | `mean(1 - won_game donde current_returner == p1_id)` |
| p1_aces_W | `mean(aces donde current_server == p1_id)` |
| p1_df_W | `mean(double_faults donde current_server == p1_id)` |
| p1_1st_pct_W | `mean(first_serve_in_pct donde current_server == p1_id)` |
| p1_svr_pts_won_W | `mean(svr_pts_won_pct donde current_server == p1_id)` |
| p1_winners_W | `mean(winners donde p1 sirve o recibe)` |
| p1_ue_W | `mean(unforced_errors donde p1 sirve o recibe)` |
| p1_fe_W | `mean(forced_errors donde p1 sirve o recibe)` |
| error_rate_W | `sum(unforced_errors + forced_errors) / sum(n_points)` |
| win_trend_W | `mean(won_game de primera mitad) - mean(won_game de segunda mitad)` |

### 2.4 Verificación de leakage (objetivamente verificable)

**Test**: `src/features/sliding_window.py::test_no_leakage()`

**Resultado: PASSED**

- Verifica que para game N, la ventana solo incluye games 1..N-1
- Verifica que el último game en ventana es N-1 (no N)
- Verifica que feature manual no usa game actual

```python
window = indicators.iloc[start:test_idx]  # test_idx = current game index
# window.index[-1] == test_idx - 1  (último game en ventana)
# test_idx NOT in window.index      (game actual excluido)
```

### 2.5 Resolución de multicolinealidad

**Problema detectado**: 2 pares de features con |r| > 0.95 en la ventana utilizada para la detección; ambos presentan correlación perfecta y se repiten en las tres ventanas.

**Features eliminadas (6)**:

| Features redundantes | Razón | Correlación |
|---|---|---|
| p2_serve_wr_W{2,3,5} | Complemento perfecto de p1_* | r = -1.0 |
| p2_return_wr_W{2,3,5} | Complemento perfecto de p1_* | r = -1.0 |

**Resultado después**: 0 pares con |r| > 0.95. Esto significa que no quedan pares por encima del umbral de detección; no significa que todas las correlaciones residuales sean iguales a cero. En la ventana W=5, el máximo residual observado es |r| = 0.831.

### 2.6 Test de significancia estadística

**Metodología**: Correlación de Spearman (no paramétrica, robusta a outliers)
- Alpha = 0.05
- Efecto mínimo = |r| > 0.01

**Resultados**:
- Features analizadas: 48
- Significativas (p < 0.05): 35
- Con efecto (|r| > 0.01): 30
- Significativas Y con efecto: 30

### 2.7 Top 10 features por correlación

| Feature | r | p-value | N |
|---|---|---|---|
| p2_svr_pts_won_W5 | -0.0737 | 5.54e-98 | 81,040 |
| p2_serve_wr_W5 | -0.0703 | 2.41e-89 | 81,040 |
| p1_return_wr_W5 | +0.0703 | 2.41e-89 | 81,040 |
| p1_svr_pts_won_W5 | +0.0699 | 3.05e-92 | 84,611 |
| p2_return_wr_W5 | -0.0671 | 5.92e-85 | 84,611 |
| p1_serve_wr_W5 | +0.0671 | 5.92e-85 | 84,611 |
| p2_svr_pts_won_W3 | -0.0658 | 1.74e-78 | 81,040 |
| p1_svr_pts_won_W3 | +0.0634 | 5.04e-76 | 84,611 |
| p1_return_wr_W3 | +0.0632 | 1.68e-72 | 81,040 |
| p2_serve_wr_W3 | -0.0632 | 1.68e-72 | 81,040 |

### 2.8 Validación train/test split (objetivamente verificable)

**Metodología**: Split temporal
- Train: años < 2023 (163,440 rows)
- Test: años >= 2023 (101,106 rows)

**Resultados**:
- Features significativas en train: 30
- Features significativas en test: 27
- **Features estables** (significativas en ambos + mismo signo): **20**
- Features inestables (difieren entre train/test): 17

**Top 5 features estables**:

| Feature | r_train | r_test |
|---|---|---|
| p2_svr_pts_won_W5 | -0.0755 | -0.0704 |
| p1_return_wr_W5 | +0.0715 | +0.0681 |
| p2_svr_pts_won_W3 | -0.0678 | -0.0622 |
| p1_svr_pts_won_W5 | +0.0665 | +0.0758 |
| p1_serve_wr_W5 | +0.0636 | +0.0731 |

### 2.9 Cobertura de features

- **Nulos esperados** (cold start): aproximadamente 4.7% en las features utilizadas por los modelos
- **Justificación**: Rolling stats requieren historial; primeros games no tienen suficiente data

---

## Resumen de cumplimiento

### P1 - Dataset limpio, normalizado y estructurado

| Indicador | Cumplido |
|---|---|
| 904,128 puntos limpios (579 duplicados removidos) | ✅ |
| 18 columnas críticas con 0% nulos | ✅ |
| 7/7 tests de integridad PASS | ✅ |
| Distribución por superficie documentada | ✅ |
| Player stats con cobertura ≥96% | ✅ |
| Target definido (96.1% coverage) | ✅ |

### P2 - Características dinámicas

| Indicador | Cumplido |
|---|---|
| 48 features útiles (6 redundantes eliminadas) | ✅ |
| Ningún par restante supera |r| = 0.95 | ✅ |
| Test de leakage PASSED | ✅ |
| 30 features significativas y con efecto (p<0.05, \|r\|>0.01) | ✅ |
| 20 features estables en train/test split | ✅ |
| Fórmulas documentadas | ✅ |

---

## Archivos generados

| Archivo | Propósito |
|---|---|
| `data/processed/dataset_clean.parquet` | Dataset punto a punto (68 cols) |
| `data/processed/matches.parquet` | Metadata del partido + player stats (21 cols) |
| `data/processed/games.parquet` | Game-level + target (32 cols) |
| `data/processed/game_features.parquet` | Features antes de resolución (57 columnas: 54 features + 3 índices) |
| `data/processed/game_features_resolved.parquet` | Features sin redundancia (51 columnas: 48 features + 3 índices) |
| `data/processed/significance_report.md` | Test de significancia |
| `data/processed/train_test_validation.md` | Validación temporal |
| `data/processed/coverage_improvement.md` | Mejora de cobertura con LOCF |
| `reports/01_calidad_datos.md` | Calidad y tests de integridad |
| `reports/02_game_dataset.md` | Descripción del dataset de games |
| `reports/data_dictionary.md` | Diccionario de datos |

## Scripts

| Script | Propósito |
|---|---|
| `src/preprocess/quality.py` | Calidad + tests de integridad |
| `src/features/sliding_window.py` | Cálculo de features + test de leakage |
| `src/features/multicollinearity.py` | Detección y resolución de multicolinealidad |
| `src/features/significance_test.py` | Test de significancia |
| `src/features/train_test_validation.py` | Validación temporal |
| `src/features/coverage_improvement.py` | LOCF para rank/age |
