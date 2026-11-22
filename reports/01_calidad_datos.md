# Reporte de Calidad de Datos (R1)

Puntos analizados: **904,128** en **5,569** partidos.

## 1. Cobertura

Distribución por superficie:

- Hard: 569,959 puntos (63.0%)
- Clay: 232,162 puntos (25.7%)
- Grass: 101,626 puntos (11.2%)
- Unknown: 381 puntos (0.0%)

## 2. Incidencias del parser

Puntos parseados sin incidencias: **99.6%**
- `ok`: 900,941 (99.6%)
- `special_point`: 2,553 (0.3%)
- `stopped_challenge`: 325 (0.0%)
- `no_ending_ret_won`: 250 (0.0%)
- `no_ending_svr_won`: 40 (0.0%)
- `unresolved_double_fault`: 19 (0.0%)

## 3. Indicadores derivados clave

- **% de puntos ganados por el sacador**: 64.26%
- **% de aces**: 8.34%
- **% de dobles faltas**: 3.41%
- **% de errores no forzados**: 32.05%
- **% de errores forzados**: 30.17%
- **% de winners de rally**: 22.98%
- **% de puntos de quiebre**: 8.06%
- **Golpes promedio por punto**: 4.10
- **1er servicio dentro (%)**: 62.79%
- **% de puntos con finalizador inferido**: 0.03%

## 4. Nulos e inconsistencias

Celdas nulas totales: **2,943,476** (4.79%)
Columnas con al menos un nulo (18): `Gm2`, `TbSet`, `special`, `serve_dir`, `serve_rank`, `ending`, `error_type`, `return_shot_type`, `return_dir`, `return_depth`, `pt_srv`, `pt_ret`, `Date`, `Time`, `Court`, `Umpire`, `Final TB?`, `Charted by`

### Cumplimiento por registro

- Campos obligatorios completos: **904,128/904,128** (**100.00%**).
- Registros que también cumplen todas las reglas: **903,936/904,128** (**99.98%**).
- Los campos opcionales pueden contener nulos. Las incidencias de decodificación se presentan por separado en la sección 2.

| Regla | Registros que incumplen |
|---|---:|
| Clave partido–punto única | 0 |
| Punto y juego positivos, enteros | 0 |
| Juegos consecutivos dentro del partido | 0 |
| Roles de saque y devolución coherentes | 192 |
| Superficie admitida | 0 |
| Aces y dobles faltas excluyentes | 0 |
| Contadores no negativos | 0 |
| Indicadores binarios | 0 |

## 5. Tests de integridad

- [PASS] `critical_col_no_nulls:match_id`: 0 nulls
- [PASS] `critical_col_no_nulls:Pt`: 0 nulls
- [PASS] `critical_col_no_nulls:Gm#`: 0 nulls
- [PASS] `critical_col_no_nulls:Set1`: 0 nulls
- [PASS] `critical_col_no_nulls:Set2`: 0 nulls
- [PASS] `critical_col_no_nulls:Player 1`: 0 nulls
- [PASS] `critical_col_no_nulls:Player 2`: 0 nulls
- [PASS] `critical_col_no_nulls:Surface`: 0 nulls
- [PASS] `critical_col_no_nulls:current_server`: 0 nulls
- [PASS] `critical_col_no_nulls:current_returner`: 0 nulls
- [PASS] `critical_col_no_nulls:isSvrWinner`: 0 nulls
- [PASS] `critical_col_no_nulls:isAce`: 0 nulls
- [PASS] `critical_col_no_nulls:isDouble`: 0 nulls
- [PASS] `critical_col_no_nulls:isRallyWinner`: 0 nulls
- [PASS] `critical_col_no_nulls:isUnforced`: 0 nulls
- [PASS] `critical_col_no_nulls:isForced`: 0 nulls
- [PASS] `critical_col_no_nulls:rallyCount`: 0 nulls
- [PASS] `critical_col_no_nulls:net_approaches`: 0 nulls
- [PASS] `game_num_contiguous`: All matches have contiguous game numbers
- [PASS] `won_game_consistent`: 636/91753 games mismatch (0.69%)
- [PASS] `date_parseable`: 192 unparseable dates (0.02%)
- [PASS] `surface_valid`: All valid
- [PASS] `ace_and_double_exclusive`: 0 points with both ace and double fault
- [PASS] `rally_count_non_negative`: 0 negative rally counts

**Resultado general: TODOS LOS TESTS PASARON**

## 6. Notas metodológicas

- Se eliminaron **579** filas duplicadas de
  `(match_id, Pt)` (artefactos de puntos repetidos por el chartista).

- Los CSV de puntos de Sackmann **no** traen las columnas derivadas (isAce, rallyCount, etc.); estas se reconstruyeron parseando `1st`/`2nd` con `src/preprocess/parser.py` (Fase 2.1).
- `special_point` (S/R/P/Q) indica puntos censados sin códigos; su ganador se toma de `PtWinner`.
- `no_ending_*` indica finalizador ausente; el resultado se infiere de `PtWinner`.
- `1stIn` se considera True si el primer servicio entró y el rally se jugó.
