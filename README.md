# TactisPro

Dashboard de investigación para explorar el momentum en partidos de tenis. Reproduce registros históricos punto a punto y actualiza el análisis cuando se cierra cada juego. La interfaz muestra el índice observado, los pronósticos del siguiente juego, los cruces confirmados y los indicadores acumulados de ambos jugadores.

Proyecto de tesis de Ingeniería Informática, Pontificia Universidad Católica del Perú.

## Arranque rápido

La portada de la aplicación necesita dos procesos locales: el servicio de inferencia Starlette y la interfaz Streamlit. El repositorio incluye la copia de los datos procesados y los resultados que permiten preparar el servicio. Se verificó con Python 3.14.

Desde la raíz del repositorio:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m backend.train
```

`backend.train` ajusta R4 y R5 con las configuraciones registradas en el experimento y guarda el paquete local de modelos. Basta ejecutarlo una vez, o después de regenerar los datos o cambiar los resultados de selección. El servicio también carga la GRU de R6 incluida en los datos procesados.

Inicia el backend en una terminal:

```bash
source .venv/bin/activate
python -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

Inicia la interfaz en otra terminal:

```bash
source .venv/bin/activate
python -m streamlit run dashboard/app.py
```

Abre [TactisPro local](http://localhost:8501). Si el servicio de inferencia usa otra dirección, configura `TACTISPRO_BACKEND_URL` en la terminal de Streamlit. Para usar datos de otra carpeta, define `DATA_DIR` con la misma ruta en ambos procesos. La carpeta debe contener el subdirectorio `processed/` con datos y modelos.

## Qué muestra el dashboard

| Resultado | Lectura en el sistema |
|---|---|
| R3 | Índice observado por juego y curvas individuales de los jugadores. |
| R4 | Valor previsto del índice para el siguiente juego. El historial compara ese valor con el observado mediante el error absoluto. |
| R5 | Probabilidad de que aumente el índice. El historial evalúa el acierto de dirección por separado. |
| R6 | Estimación de juegos hasta el siguiente cruce, condicionada a que se observe uno antes del final. |
| R8–R10 | Reproducción histórica, señales de R5 en lenguaje de tenis y alertas por cruces observados del índice. |

R7 es el prototipo de diseño con datos simulados. Se puede abrir por separado desde [dashboard/mockup/index.html](dashboard/mockup/index.html).

## Estructura

| Carpeta | Contenido |
|---|---|
| [dashboard/](dashboard/) | Interfaz, visualizaciones y cliente HTTP. |
| [backend/](backend/) | Preparación de modelos y servicio de inferencia por juego. |
| [src/](src/) | Preparación de datos, variables, índice, entrenamiento y evaluación. |
| [data/](data/) | Copia de los insumos de investigación y resultados reproducibles. |
| [reports/](reports/) | Reportes de calidad, modelado, alertas y protocolo de usabilidad. |
| [tests/](tests/) | Pruebas de temporalidad y detección de cruces. |

## Reproducir la evaluación

Para regenerar las salidas desde la copia de los CSV, ejecuta los módulos en este orden:

```bash
python -m src.preprocess.clean
python -m src.preprocess.quality
python -m scripts.build_game_dataset
python -m src.features.multicollinearity
python -m src.features.significance_test
python -m src.features.train_test_validation
python -m src.features.predictive_contribution
python -m src.momentum.engine
python -m src.momentum.validate
python -m src.momentum.select_window
python -m src.momentum.regression
python -m src.momentum.classification
python -m src.momentum.crossing_forecast
python -m backend.train
python -m src.momentum.permutation_audit
python -m src.momentum.predictive_audit
python -m src.momentum.crossing_diagnostics
python -m src.momentum.evaluate_alerts
python -m unittest discover -s tests
```

La reproducción completa vuelve a escribir resultados y puede tardar. Las figuras se generan localmente en `docs_tesis/figures/`. Para revisar la parte técnica y el procedimiento de latencia, consulta la [guía del dashboard](dashboard/README.md).

## Datos y atribución

La fuente punto a punto es el [Match Charting Project de Tennis Abstract y Jeff Sackmann](https://github.com/JeffSackmann/tennis_MatchChartingProject). Los datos complementarios de partidos ATP proceden de [tennis_atp de Jeff Sackmann](https://github.com/JeffSackmann/tennis_atp). Las fuentes publican estos datos bajo [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Las tablas procesadas derivan de esos registros mediante limpieza, agregación por juego y cálculo de variables e índices.
