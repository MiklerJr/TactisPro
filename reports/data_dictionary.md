# Diccionario de datos

Dataset: `data/processed/dataset_clean.parquet` (punto a punto).

## Columnas críticas (0% nulos requerido)

| Campo | Descripción |
|---|---|
| `match_id` | Identificador único del partido. |
| `Pt` | Número de punto dentro del partido, desde 1. |
| `Gm#` | Número de juego del partido, desde 1. |
| `Set1` | Sets ganados por el Player 1 hasta el momento. |
| `Set2` | Sets ganados por el Player 2 hasta el momento. |
| `Player 1` | Nombre del jugador 1 (definido por el primer servicio). |
| `Player 2` | Nombre del jugador 2. |
| `Surface` | Superficie: Hard, Clay, Grass, Unknown. |
| `current_server` | Nombre del jugador que sirve en este punto. |
| `current_returner` | Nombre del jugador que recibe en este punto. |
| `isSvrWinner` | 1 si ganó el punto el sacador. |
| `isAce` | True si el punto terminó en ace. |
| `isDouble` | True si el punto fue doble falta. |
| `isRallyWinner` | True si el punto terminó en winner (excluye aces). |
| `isUnforced` | True si el punto terminó en error no forzado. |
| `isForced` | True si el punto terminó en error forzado. |
| `rallyCount` | Número de golpes del rally (incluye el saque). |
| `net_approaches` | Número de subidas a la red del sacador. |

## Columnas opcionales (aceptan nulos)

| Campo | Descripción |
|---|---|
| `Pts` | Marcador de puntos del juego actual. |
| `TbSet` | True si el set se decide por tiebreak. |
| `Svr` | 1 si el Player 1 sirve, 2 si sirve el Player 2. |
| `1st` | Código del primer servicio. |
| `2nd` | Código del segundo servicio. |
| `Notes` | Notas del chartista. |
| `PtWinner` | 1 si ganó el Player 1, 2 si ganó el Player 2. |
| `lets_1st` | Lets en el primer servicio. |
| `lets_2nd` | Lets en el segundo servicio. |
| `special` | Código especial: S/R (punto cedido), P/Q (penalización). |
| `1stIn` | True si el primer servicio entró. |
| `2ndIn` | True si el segundo servicio entró. |
| `1stSV` | True si el primer servicio fue serve-and-volley. |
| `2ndSV` | True si el segundo servicio fue serve-and-volley. |
| `serve_dir` | Dirección del servicio: wide, body, t, unknown. |
| `serve_rank` | Servicio en juego: 1 o 2. |
| `isUnret` | True si el restador no devolvió el saque. |
| `ending` | Finalizador: winner, forced, unforced. |
| `error_type` | Tipo de error final. |
| `return_shot_type` | Tipo de golpe de la devolución. |
| `return_dir` | Dirección de la devolución. |
| `return_depth` | Profundidad de la devolución. |
| `n_shots` | Número de golpes registrados tras el saque. |
| `parse_issue` | Incidencia del parser: ok, special_point, etc. |
| `ending_inferred` | 1 si el finalizador se infirió desde PtWinner. |
| `pts_srv_raw` | Componente del marcador del sacador. |
| `pts_ret_raw` | Componente del marcador del restador. |
| `is_tiebreak` | True si el punto pertenece a un tiebreak. |
| `pt_srv` | Índice de puntos del sacador (0/1/2/3/4+). |
| `pt_ret` | Índice de puntos del restador. |
| `deuce` | True si el juego está en deuce. |
| `ad_srv` | True si el sacador tiene ventaja. |
| `ad_ret` | True si el restador tiene ventaja (break point). |
| `is_break_point` | True si el punto es de quiebre. |
| `current_server_hand` | Mano del sacador (R/L). |
| `current_returner_hand` | Mano del restador (R/L). |
| `p1_player_id` | ID numérico único del Player 1. |
| `p2_player_id` | ID numérico único del Player 2. |
| `isRtrWinner` | 1 si ganó el punto el restador. |
| `Pl 1 hand` | Mano del jugador 1. |
| `Pl 2 hand` | Mano del jugador 2. |
| `Date` | Fecha del partido (datetime). |
| `Tournament` | Torneo. |
| `Round` | Ronda. |
| `Court` | Pista. |
| `Umpire` | Árbitro. |
| `Best of` | Partido a 3 o 5 sets. |
| `Final TB?` | Regla del set final para el tiebreak. |
| `Charted by` | Chartista que registró el partido. |