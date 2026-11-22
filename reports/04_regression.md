# R4. MODELO BASE DE REGRESION - PREDICCION DE MOMENTUM

## 4.1 Objetivo

Predecir el valor numerico del indice de momentum en el siguiente game
(t+1) usando tres algoritmos de regresion: Random Forest, XGBoost y SVR.
El mejor modelo se selecciona como base para el clasificador de la direccion del cambio (R5).

## 4.2 Configuracion experimental

**Datos de entrada**:
- Features dinamicas R2 (W=5): 16 columnas.
- Features autorregresivas: M_A(t), M_B(t), N(t).
- Total features: 19.
- Filas: 1 game = 1 fila (se filtra window_size=5; el parquet trae 3
  filas por game, una por ventana).

**Target**: `N_next = N(t+1)` (primario); `M_A_next`, `M_B_next` (secundarios).

**Split temporal**:
- Train: anos < 2023.
- Test: anos >= 2023.
- **Sin leakage**: el target esta shifted al siguiente game del mismo partido.

**Hiperparametros**:

| Modelo | Hiperparametros |
|---|---|
| Random Forest | n_estimators=50, max_depth=6, random_state=42 |
| XGBoost | n_estimators=100, max_depth=4, lr=0.1, subsample=0.8, colsample=0.8 |
| SVR | kernel=rbf, C=1.0, scaler=StandardScaler (subset 5k filas, O(n^2)) |

**Validacion**: Tres folds cronológicos por fecha completa, con imputación ajustada en cada fold.

## 4.3 Resultados en test set

**Tabla 1. Metricas en test (MAE, RMSE, R^2)**

| Modelo | Target | MAE | RMSE | R^2 |
|---|---|---|---|---|
| Random Forest | N | 0.2160 | 0.3291 | 0.6827 |
| Random Forest | M_A | 0.1266 | 0.2021 | 0.7175 |
| Random Forest | M_B | 0.1249 | 0.2029 | 0.7221 |
| XGBoost | N | 0.2116 | 0.3221 | 0.6961 |
| XGBoost | M_A | 0.1253 | 0.2001 | 0.7231 |
| XGBoost | M_B | 0.1236 | 0.1993 | 0.7321 |
| SVR | N | 0.2246 | 0.3442 | 0.6531 |
| SVR | M_A | 0.1464 | 0.2193 | 0.6673 |
| SVR | M_B | 0.1426 | 0.2171 | 0.6820 |

**Tabla 2. Baselines ingenuos (mismo test set)**

| Baseline | Target | MAE | RMSE | R^2 |
|---|---|---|---|---|
| Persistencia (t -> t+1) | N | 0.1994 | 0.3509 | 0.6394 |
| Media del train | N | 0.4723 | 0.5850 | -0.0025 |
| Persistencia (t -> t+1) | M_A | 0.1283 | 0.2455 | 0.5833 |
| Media del train | M_A | 0.3443 | 0.3806 | -0.0020 |
| Persistencia (t -> t+1) | M_B | 0.1221 | 0.2419 | 0.6049 |
| Media del train | M_B | 0.3509 | 0.3851 | -0.0010 |

**Tabla 3. Validacion cruzada 3-fold (TimeSeriesSplit)**

| Modelo | Target | CV MAE (media ± DE) | CV RMSE (media ± DE) | CV R² (media ± DE) |
|---|---|---|---|---|
| Random Forest | N | 0.2172 ± 0.0014 | 0.3328 ± 0.0037 | 0.6708 ± 0.0052 |
| Random Forest | M_A | 0.1275 ± 0.0021 | 0.2040 ± 0.0046 | 0.7088 ± 0.0108 |
| Random Forest | M_B | 0.1279 ± 0.0008 | 0.2083 ± 0.0016 | 0.7078 ± 0.0043 |
| XGBoost | N | 0.2158 ± 0.0025 | 0.3273 ± 0.0044 | 0.6816 ± 0.0066 |
| XGBoost | M_A | 0.1272 ± 0.0027 | 0.2023 ± 0.0050 | 0.7137 ± 0.0118 |
| XGBoost | M_B | 0.1281 ± 0.0004 | 0.2056 ± 0.0020 | 0.7153 ± 0.0054 |
| SVR | N | 0.2289 ± 0.0041 | 0.3467 ± 0.0035 | 0.6427 ± 0.0051 |
| SVR | M_A | 0.1505 ± 0.0037 | 0.2205 ± 0.0049 | 0.6598 ± 0.0125 |
| SVR | M_B | 0.1503 ± 0.0012 | 0.2235 ± 0.0016 | 0.6637 ± 0.0042 |

![Comparativa](figures/11_regression_comparison.png)

## 4.4 Seleccion del mejor modelo

**Modelo seleccionado (target primario N_next)**: **XGBoost**

- MAE = 0.2116
- RMSE = 0.3221
- R^2 = 0.6961

**Criterio de seleccion**: menor MAE medio en validación cruzada sobre el target primario `N_next`.

## 4.5 Cumplimiento del indicador R4

**Indicador del EDT**: 'Comparativa de Random Forest, XGBoost y SVR mediante
MAE, RMSE y R^2 en validacion cruzada. El mejor modelo se selecciona como base.'

**Resultado**:
- Se entrenaron y compararon los 3 algoritmos requeridos.
- Se reportan MAE, RMSE y R^2.
- Se aplico validacion cruzada temporal (3 folds por fecha completa).
- Se selecciono el modelo de minimo MAE como base.

**Conclusion**: el indicador R4 se cumple completamente.

## 4.6 Medio de verificacion

- Script: `src/momentum/regression.py`
- Resultados crudos: `data/processed/r4_results.json`
- Reporte: `reports/04_regression.md`
- Figura: `docs_tesis/figures/11_regression_comparison.png`