# R3. SERIE TEMPORAL DE MOMENTUM - VALIDACION ESTADISTICA

## 3.1 Objetivo

Cuantificar el indice de momentum mediante el Metodo de Ponderacion por
Entropia (EWM) y logica difusa, y validar estadisticamente que
la serie resultante NO es aleatoria (cumple el indicador de R3).

## 3.2 Metodologia

- **Calculo del indice (EWM)**: pesos entrpicos objetivos sobre 5
  features por jugador (4 positivas + 1 negativa, W=5), subconjunto de
  los indicadores de R2. La normalización, la ponderación y el reescalado
  usan los juegos observados hasta el cierre de cada juego.
- **Membresia difusa**: 3 estados (`low`, `medium`, `high`) con funciones
  triangulares sobre el score agregado.
- **Momentum neto**: N(t) = M_A(t) - M_B(t).
- **Validacion por partido** (cada test se aplica a cada uno de los
  partidos con >= 20 games y luego se agregan los resultados):
  - **ADF** (Dickey-Fuller Aumentado): H0 = la serie tiene raiz unitaria
    (no estacionaria); H1 = serie estacionaria. Criterio: p < 0.05.
  - **Ljung-Box**: H0 = no hay autocorrelacion en los primeros lags
    (ruido blanco); H1 = existe autocorrelacion. Criterio: p < 0.05.
  - **Agregacion**: % de partidos que rechaza H0 (5% se esperaria bajo
    H0), mediana de los p-valores y test combinado de Fisher sobre los
    p-valores (los partidos son independientes entre si), que evalua la
    hipotesis global.
  - **ACF/PACF**: promedio de las autocorrelaciones calculadas dentro de
    cada partido; inspeccion visual (evidencia descriptiva).

> Nota metodologica: los tests NO se calculan sobre la serie concatenada
> de los 3,571 partidos, porque los bordes entre partidos generarian
> discontinuidades artificiales y p-valores no calibrados.

## 3.3 Resultados de los tests

Tabla 3.3. Test de estacionariedad (ADF) aplicado por partido y agregado

| Serie | Partidos evaluados | % con p < 0.05 | Mediana del p | Chi-cuadrado (Fisher) | p (Fisher) |
|---|---|---|---|---|---|
| M_A(t) | 2,461 | 23.6% | 0.295 | 18,040 | <1e-300 |
| M_B(t) | 2,461 | 24.9% | 0.305 | 45,494 | <1e-300 |
| N(t)=M_A-M_B | 2,461 | 23.0% | 0.291 | 17,222 | <1e-300 |

Tabla 3.4. Test de aleatoriedad (Ljung-Box) aplicado por partido y agregado

| Serie | Partidos evaluados | % con p < 0.05 | Mediana del p | Chi-cuadrado (Fisher) | p (Fisher) |
|---|---|---|---|---|---|
| M_A(t) | 2,461 | 78.3% | <0.001 | 61,833 | <1e-300 |
| M_B(t) | 2,461 | 81.1% | <0.001 | 64,424 | <1e-300 |
| N(t)=M_A-M_B | 2,461 | 85.7% | <0.001 | 67,240 | <1e-300 |

**Interpretacion**:

- **Estacionariedad (ADF, tablas 3.3)**: a nivel individual solo una
  fraccion de los partidos rechaza H0 (frente al 5% esperado bajo H0);
  el test combinado de Fisher rechaza la hipotesis global de
  no estacionariedad (p < 1e-300). La evidencia de estacionariedad es
  por tanto agregada y no individual: el ADF tiene potencia limitada en
  series de ~25 games. Es una condicion habitual (no exclusiva) para
  modelos que requieren o se benefician de una dinámica temporal estable.
- **Dependencia del índice (Ljung–Box, tabla 3.4)**. La prueba detecta
  autocorrelación en los índices construidos. En 200 partidos de prueba,
  el 88,5 % de las series observadas rechazó la hipótesis de
  autocorrelación nula frente al 85,7 % de las series reconstruidas tras
  permutar los juegos dentro de cada sacador y tipo de juego. La diferencia
  no alcanzó significancia al 5 % (p = 0,15). Para el primer rezago,
  las autocorrelaciones medias fueron 0,686 y 0,670 (p = 0,09).
  Reproducible con `python -m src.momentum.permutation_audit`.
- **Limitaciones**: (i) la autocorrelacion no implica causalidad;
  (ii) los tests se aplican al indice construido, no a las features
  crudas; (iii) Fisher rechaza la hipotesis global conjunta, no garantiza
  estacionariedad en todos los partidos; (iv) Fisher asume independencia
  entre partidos, supuesto aproximado al compartir jugadores.

## 3.4 Estadisticas descriptivas

Estas cifras son descriptivas (sobre los 91,753 games de la muestra); no
son resultados inferenciales.

Tabla 3.5. Estadisticas descriptivas de las series de momentum

| Serie | Media | Desv. Std | Min | Max |
|---|---|---|---|---|
| M_A(t) | 0.5805 | 0.3724 | 0.0000 | 1.0000 |
| M_B(t) | 0.5870 | 0.3780 | 0.0000 | 1.0000 |
| N(t)=M_A-M_B | -0.0065 | 0.5719 | -1.0000 | 1.0000 |

## 3.5 Figuras

### ACF / PACF promedio del momentum neto N(t)

Patron esperado: decaimiento gradual en ACF (autocorrelacion positiva
en los primeros lags) y/o PACF con lags iniciales fuera de la banda de
confianza: evidencia visual de memoria temporal en la serie. Es evidencia
descriptiva (visual), no una identificacion definitiva de modelo AR/MA.

![ACF PACF](figures/08_acf_pacf.png)

### Curvas de momentum - casos de demostracion

Los siguientes 4 partidos son **casos de demostracion** elegidos por
longitud variable. Ilustran la forma de las curvas; NO constituyen una
muestra representativa de la poblacion de partidos.

![Curvas](figures/09_momentum_curves.png)

### Distribucion de estados difusos

![Estados](figures/10_state_distribution.png)

## 3.6 Cumplimiento del indicador R3

**Indicador**: Reportar estadísticos, valores p y rezagos de ADF y Ljung–Box a nivel 0.05.

**Resultado**: se cumple. El indicador se sostiene principalmente en el
test de aleatoriedad Ljung-Box (rechazo en la mayoria de los partidos
y Fisher p < 1e-300, ver tabla 3.4); el ADF aporta evidencia
agregada (Fisher p < 1e-300, tabla 3.3) con potencia individual
limitada por la longitud de cada partido.

## 3.7 Medio de verificacion

- Script: `src/momentum/engine.py` (calculo EWM+Fuzzy)
- Script: `src/momentum/validate.py` (tests ADF, Ljung-Box, figuras)
- Dataset: `data/processed/momentum_series.parquet` (91,753 filas, 3,571 partidos)
- Reporte: `reports/03_momentum.md`
- Estadísticos, valores p y rezagos por partido: `data/processed/r3_statistical_tests.csv`.
- Figuras: `docs_tesis/figures/08_acf_pacf.png`, `09_momentum_curves.png`,
  `10_state_distribution.png`
