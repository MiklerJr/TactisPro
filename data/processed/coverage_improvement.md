# Mejora de Cobertura de Player Stats

**Fecha**: 2026-09-15 10:33

## Estrategia

Para jugadores sin rank/age en Sackmann, se usa la ultima observacion
disponible (LOCF - Last Observation Carried Forward), limitada a 3 anos.

## Comparacion de cobertura

| Campo | Antes | Despues | Mejora |
|---|---|---|---|
| p1_rank | 83.3% | 96.3% | +13.0% |
| p2_rank | 83.0% | 96.6% | +13.6% |
| p1_age | 83.3% | 96.3% | +13.0% |
| p2_age | 83.0% | 96.6% | +13.6% |

## Cobertura por ano (despues)

| Ano | Matches | p1_rank cov |
|---|---|---|
| 2010 | 69 | 100.0% |
| 2011 | 75 | 96.0% |
| 2012 | 86 | 96.5% |
| 2013 | 88 | 97.7% |
| 2014 | 102 | 94.1% |
| 2015 | 181 | 95.0% |
| 2016 | 124 | 95.2% |
| 2017 | 148 | 96.6% |
| 2018 | 158 | 98.1% |
| 2019 | 384 | 96.6% |
| 2020 | 144 | 98.6% |
| 2021 | 249 | 99.2% |
| 2022 | 404 | 98.8% |
| 2023 | 387 | 100.0% |
| 2024 | 470 | 96.6% |
| 2025 | 361 | 90.3% |
| 2026 | 141 | 85.1% |

## Metodologia

- **LOCF (Last Observation Carried Forward)**: usa la ultima rank/age conocida
- **Ventana**: maximo 3 anos entre match y ultima observacion
- **Justificacion**: jugadores activos mantienen rank similar ano a ano
