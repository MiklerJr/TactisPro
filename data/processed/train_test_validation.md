# Validacion Train/Test de Features

**Fecha**: 2026-11-21 12:07

**Split year**: 2023
**Train**: years < 2023
**Test**: years >= 2023
**Alpha**: 0.05
**Efecto minimo**: |r| > 0.01

## Resumen

- Features totales: 48
- Significativas en train: 30
- Significativas en test: 27
- Estables (significativas en ambos + mismo signo): 20
- Inestables (difieren entre train/test): 17

## Features Estables (significativas en train Y test)

| Feature | r_train | p_train | r_test | p_test | N_train | N_test |
|---|---|---|---|---|---|---|
| p2_svr_pts_won_W5 | -0.0755 | 2.98e-64 | -0.0704 | 2.67e-35 | 50,056 | 30,984 |
| p1_return_wr_W5 | +0.0715 | 1.07e-57 | +0.0681 | 3.62e-33 | 50,056 | 30,984 |
| p2_svr_pts_won_W3 | -0.0678 | 4.43e-52 | -0.0622 | 6.44e-28 | 50,056 | 30,984 |
| p1_svr_pts_won_W5 | +0.0665 | 2.84e-52 | +0.0758 | 2.01e-42 | 52,268 | 32,343 |
| p1_serve_wr_W5 | +0.0636 | 6.07e-48 | +0.0731 | 1.49e-39 | 52,268 | 32,343 |
| p1_return_wr_W3 | +0.0630 | 2.88e-45 | +0.0631 | 9.41e-29 | 50,056 | 30,984 |
| p1_svr_pts_won_W3 | +0.0616 | 4.65e-45 | +0.0665 | 4.85e-33 | 52,268 | 32,343 |
| p1_serve_wr_W3 | +0.0590 | 1.65e-41 | +0.0654 | 5.68e-32 | 52,268 | 32,343 |
| p2_svr_pts_won_W2 | -0.0578 | 2.93e-38 | -0.0610 | 6.25e-27 | 50,056 | 30,984 |
| p1_return_wr_W2 | +0.0548 | 1.42e-34 | +0.0619 | 1.12e-27 | 50,056 | 30,984 |
| p1_svr_pts_won_W2 | +0.0544 | 1.48e-35 | +0.0562 | 4.35e-24 | 52,268 | 32,343 |
| p1_serve_wr_W2 | +0.0521 | 1.01e-32 | +0.0571 | 8.48e-25 | 52,268 | 32,343 |
| p2_aces_W5 | -0.0374 | 5.46e-17 | -0.0267 | 2.60e-06 | 50,056 | 30,984 |
| p2_aces_W3 | -0.0335 | 6.31e-14 | -0.0218 | 1.28e-04 | 50,056 | 30,984 |
| p2_aces_W2 | -0.0280 | 3.53e-10 | -0.0217 | 1.37e-04 | 50,056 | 30,984 |
| p1_aces_W5 | +0.0242 | 3.03e-08 | +0.0462 | 1.01e-16 | 52,268 | 32,343 |
| p1_aces_W3 | +0.0214 | 1.03e-06 | +0.0411 | 1.52e-13 | 52,268 | 32,343 |
| p1_aces_W2 | +0.0186 | 2.13e-05 | +0.0350 | 3.12e-10 | 52,268 | 32,343 |
| p1_df_W5 | -0.0185 | 2.42e-05 | -0.0143 | 9.90e-03 | 52,268 | 32,343 |
| p1_1st_pct_W5 | +0.0184 | 2.63e-05 | +0.0136 | 1.45e-02 | 52,268 | 32,343 |

## Features Inestables (significativas en uno solo)

| Feature | r_train | p_train | r_test | p_test |
|---|---|---|---|---|
| p1_df_W2 | -0.0156 | 3.51e-04 | -0.0108 | 5.10e-02 |
| p1_1st_pct_W2 | +0.0129 | 3.14e-03 | +0.0068 | 2.20e-01 |
| p2_df_W2 | +0.0155 | 5.25e-04 | +0.0082 | 1.49e-01 |
| p2_1st_pct_W2 | -0.0177 | 7.22e-05 | -0.0017 | 7.64e-01 |
| ue_W2 | -0.0041 | 3.52e-01 | -0.0144 | 9.36e-03 |
| win_trend_W2 | +0.0049 | 2.73e-01 | +0.0122 | 3.18e-02 |
| p1_df_W3 | -0.0177 | 5.37e-05 | -0.0109 | 5.03e-02 |
| p1_1st_pct_W3 | +0.0161 | 2.30e-04 | +0.0089 | 1.09e-01 |
| p2_df_W3 | +0.0176 | 7.95e-05 | +0.0098 | 8.30e-02 |
| p2_1st_pct_W3 | -0.0208 | 3.24e-06 | -0.0012 | 8.38e-01 |
| winners_W3 | -0.0012 | 7.87e-01 | -0.0115 | 3.80e-02 |
| ue_W3 | -0.0036 | 4.15e-01 | -0.0179 | 1.29e-03 |
| win_trend_W3 | +0.0018 | 6.93e-01 | +0.0145 | 1.06e-02 |
| p2_df_W5 | +0.0182 | 4.80e-05 | +0.0083 | 1.42e-01 |
| p2_1st_pct_W5 | -0.0207 | 3.59e-06 | +0.0030 | 5.93e-01 |
| winners_W5 | -0.0041 | 3.45e-01 | -0.0124 | 2.57e-02 |
| ue_W5 | -0.0028 | 5.17e-01 | -0.0163 | 3.42e-03 |

## Conclusiones

- **20** features son robustas (significativas y con efecto en train y test)
- **17** features son inestables (pueden ser ruido o spurias)
- Features estables son candidatas para el modelo final
