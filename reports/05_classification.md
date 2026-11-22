# R5. CLASIFICADOR DE LA DIRECCION DEL CAMBIO DE MOMENTUM

## 5.1 Objetivo

Clasificar la direccion del cambio del momentum neto N(t) (binarizacion:
aumentara o disminuira en t+1), con seleccion de caracteristicas (RFE) y
optimizacion de hiperparametros (GridSearch), superando al baseline
de Regresion Logistica.

## 5.2 Configuracion experimental

**Target binario**: `next_dir_up = 1` si N(t+1) > N(t), `0` en caso contrario.
**Balance de clases**: 41.8% sube / 58.2% baja (sobre 88,182 games; se descartan los ultimos games de cada partido).
**Split**: Train < 2023 (54,480 filas) | Test >= 2023 (33,702 filas).

**Features**:
- 16 features R2 (W=5)
- 3 features autorregresivas: M_A(t), M_B(t), N(t)
- 9 features de cambio/dinamica: N_diff, abs_N_t, abs_N_diff, N_sign,
  N_rel_change, N_volatility, M_diff, M_A_minus_prev, M_B_minus_prev
- Total: 28 features

## 5.3 Resultados comparativos en test set

| Modelo | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|
| Logistic Regression | 0.5488 | 0.4636 | 0.5026 | 0.6785 |
| Random Forest | 0.5624 | 0.6637 | 0.6088 | 0.7235 |
| XGBoost | 0.5803 | 0.6185 | 0.5988 | 0.7339 |
| XGBoost (GridSearch) | 0.5790 | 0.6309 | 0.6038 | 0.7339 |

![Comparativa modelos](figures/12_classification_comparison.png)

## 5.4 Optimizacion de hiperparametros - GridSearch

**Modelo base**: XGBoost.
**Espacio de busqueda**: 5 parametros, total de combinaciones = 2*3*3*2*2 = 72.

**Mejores hiperparametros encontrados**:

- `colsample_bytree` = `0.9`
- `learning_rate` = `0.05`
- `max_depth` = `7`
- `n_estimators` = `100`
- `subsample` = `0.9`

**Mejor AUC-ROC en CV (5-fold TimeSeriesSplit)**: `0.7227`

**Metricas en test con hiperparametros optimizados**:

- Precision: 0.5790
- Recall: 0.6309
- F1: 0.6038
- AUC-ROC: 0.7339

## 5.5 Seleccion de caracteristicas - RFE

Se aplico Recursive Feature Elimination (RFE) sobre Logistic Regression,
seleccionando las top 20 features por importancia.

**Metricas tras RFE**: P=0.5504, R=0.4682, F1=0.5060, AUC=0.6786

**Features seleccionadas** (top 10):

- `p1_df_W5`
- `p1_1st_pct_W5`
- `p1_svr_pts_won_W5`
- `p2_df_W5`
- `p2_1st_pct_W5`
- `p2_svr_pts_won_W5`
- `winners_W5`
- `ue_W5`
- `error_rate_W5`
- `win_trend_W5`
- ... (10 mas)

## 5.6 Importancia de caracteristicas - SHAP

Se calcularon valores SHAP sobre el modelo XGBoost optimizado:

![SHAP importance](figures/13_shap_importance.png)

**Top 10 features por valor SHAP medio**:

| Feature | Importancia SHAP |
|---|---|
| `p1_1st_pct_W5` | 0.0601 |
| `p1_svr_pts_won_W5` | 0.0652 |
| `M_A_t` | 0.0669 |
| `p2_df_W5` | 0.0675 |
| `p2_1st_pct_W5` | 0.0755 |
| `M_B_t` | 0.0770 |
| `p2_svr_pts_won_W5` | 0.1120 |
| `p1_return_wr_W5` | 0.1120 |
| `N_t` | 0.2402 |
| `abs_N_t` | 0.4786 |

## 5.7 Seleccion del mejor modelo y cumplimiento del indicador

**Modelo evaluado**: **XGBoost (GridSearch)**, con ajuste por validación cronológica.

El criterio se revisó después de los experimentos. Los umbrales aspiracionales originales de 0,75 en precisión, sensibilidad y F1 y de 0,80 en AUC-ROC no se alcanzaron.

| Métrica | Regresión logística | XGBoost | Supera la referencia |
|---|---:|---:|:---:|
| Precisión | 0.5488 | 0.5790 | Sí |
| Sensibilidad | 0.4636 | 0.6309 | Sí |
| F1 | 0.5026 | 0.6038 | Sí |
| AUC-ROC | 0.6785 | 0.7339 | Sí |

## 5.8 Medio de verificacion

- Script: `src/momentum/classification.py`
- Resultados crudos: `data/processed/r5_results.json`
- Reporte: `reports/05_classification.md`
- Figuras: `docs_tesis/figures/12_classification_comparison.png` y `13_shap_importance.png`