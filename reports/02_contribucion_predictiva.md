# R2 · Contribución predictiva de las variables dinámicas

Objetivo: predecir si el jugador A gana el siguiente juego. Se compara la misma regresión logística usando el saque previsto y agregando las 16 variables dinámicas de cada ventana. La imputación y el escalado se ajustan en entrenamiento.

Entrenamiento: partidos anteriores a 2023. Prueba: partidos desde 2023. Dentro de cada ventana, ambos modelos se evalúan sobre las mismas observaciones.

| W | Entrenamiento | Prueba | Entradas | AUC-ROC | Brier | Exactitud |
|---:|---:|---:|---|---:|---:|---:|
| 2 | 54,480 | 33,702 | Saque previsto | 0.8141 | 0.1514 | 0.8142 |
| 2 | 54,480 | 33,702 | Saque previsto y variables dinámicas | 0.8366 | 0.1497 | 0.8142 |
| 3 | 54,480 | 33,702 | Saque previsto | 0.8141 | 0.1514 | 0.8142 |
| 3 | 54,480 | 33,702 | Saque previsto y variables dinámicas | 0.8382 | 0.1495 | 0.8142 |
| 5 | 54,480 | 33,702 | Saque previsto | 0.8141 | 0.1514 | 0.8142 |
| 5 | 54,480 | 33,702 | Saque previsto y variables dinámicas | 0.8411 | 0.1491 | 0.8142 |

El contraste mide el aporte conjunto de las variables para este objetivo y modelo.
