# R10 · Evaluación de las alertas de cruce por cero

Se reprodujeron partidos históricos del conjunto de prueba (2023 en adelante). El evaluador recorre los cierres de juego del registro punto a punto, aplica el mismo cálculo que utiliza el dashboard sobre el prefijo revelado y registra cada cruce una vez.

La referencia se calcula retrospectivamente sobre la serie registrada, comparando pares consecutivos de valores distintos de cero. La coincidencia exige el mismo partido, juego y dirección. Una dirección equivocada cuenta como falso positivo para la alerta emitida y falso negativo para el cruce observado.

Partidos evaluados: **1,359**. Juegos cerrados: **35,061**.

| Cruces observados | Alertas emitidas | VP | FP | FN | Precisión | Sensibilidad | F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3,698 | 3,698 | 3,698 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 |

El experimento mide la concordancia de un detector de cruces observados. El error del pronóstico de futuros cruces se evalúa por separado en R6.

Registro por evento: `data/processed/r10_alert_events.csv`.
Registro por cierre en el evaluador del replay: `data/processed/r8_replay_closures.csv`.
