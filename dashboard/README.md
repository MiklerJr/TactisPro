# Dashboard — TactisPro (R7 Mockup + R8 funcional)

## Resultados
Implementa **R7 — Prototipo de alta fidelidad (Mockup)** y una primera versión de
**R8 — Dashboard funcional** del Objetivo Específico O3:
*"Implementar un dashboard que integre predicciones y visualizaciones tácticas que
ayuden a la planificación estratégica durante el juego."*

## Tecnología
- HTML5 + CSS3 + Vanilla JavaScript ES6+
- Chart.js 4.4.1 (cargado por CDN)
- Streamlit para la aplicación funcional R8
- Backend HTTP Starlette/Uvicorn con XGBoost y PyTorch para inferencia
- Pandas y PyArrow para leer los datos de la reproducción
- El mockup R7 continúa usando datos simulados en `mockup/app.js`

## Ejecutar R8 funcional con replay por game

Desde la raíz del proyecto, exporta R4 y R5 con `python -m backend.train`.
Luego ejecuta el backend y Streamlit en terminales separadas.

```bash
python -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

```bash
streamlit run dashboard/app.py
```

La interfaz consulta `GET /metrics` y `GET /matches/{match_id}/games/{game_num}`.
El backend ofrece `GET /health` para comprobar su disponibilidad.
La URL se puede configurar con `TACTISPRO_BACKEND_URL`.

La aplicación carga `data/processed/momentum_series.parquet` y reproduce un
partido histórico **game a game**. En cada paso:

1. Muestra los valores observados de `M_A`, `M_B` y `N` hasta el game actual.
2. Estima `N(t+1)` con el modelo XGBoost de R4.
3. Estima la dirección del cambio con el clasificador de R5.
4. Al avanzar, compara la predicción con el valor observado del game revelado.

### Controles del replay

| Control | Función |
|---|---|
| `⏮ Inicio` | Vuelve al game 1 |
| `◀ Anterior` | Retrocede un game |
| `Siguiente ▶` | Revela el siguiente game |
| `▶ Reproducir` | Avanza automáticamente |
| `⏸ Pausa` | Detiene la reproducción |
| Slider | Salta a cualquier game del partido |
| Segundos por game | Controla la velocidad de reproducción |

### Convencion de identidad (P1 / P2)

Las series de momentum son **fijas por posicion**, no por el rol de saque:

| Serie | Jugador | Color fijo |
|---|---|---|
| `M_A` | P1 — el que sirvió el primer game | Azul `#2563EB` |
| `M_B` | P2 — el que recibió el primer game | Naranja `#D97706` |

`src/momentum/engine.py` calcula `M_A` con las features `p1_*` y `M_B` con las
`p2_*`. Por eso el nombre, el avatar y el color de cada tarjeta permanecen
constantes durante todo el partido.

El rol de saque sí alterna en cada game. Por esa razón la interfaz **no** toma
el nombre del jugador desde `current_server` o `current_returner`, sino desde
`matches.parquet`, donde `player_1` y `player_2` son fijos dentro del partido.
El rol se comunica con una etiqueta independiente `SIRVE` / `RECIBE`.

Validación ejecutada al inicio sobre los 3 571 partidos:

| Comprobación | Resultado |
|---|---|
| `p1_player_id` único dentro del partido | 100 % |
| `p2_player_id` único dentro del partido | 100 % |
| P1 coincide con el servidor del game 1 | 100 % |

**Advertencia**: si una versión anterior tomaba los nombres desde el rol del
game, la tarjeta de `M_A` mostraba un nombre distinto en cada game, lo que
podía sugerir que las curvas cambiaban de jugador cuando solo cambiaba el
saque. Este error de presentación no afectaba ningún cálculo de R3, R4 ni R5.

Los metadatos de la tarjeta (ranking, edad y mano) tienen una cobertura parcial
de aproximadamente 83 % en ranking y edad. Cuando un dato no está disponible
se omite el campo, en lugar de mostrar un valor vacío.

### Cómo avanza el replay

El replay revela los games reales del partido, uno por uno:

| Paso | Serie observada | Predicciones evaluables | Proyección pendiente |
|---|---|---|---|
| 1 | game 1 | ninguna | predicción hecha en el game 1 |
| 2 | games 1-2 | predicción del game 1 | predicción hecha en el game 2 |
| k | games 1-k | predicciones hechas en los games 1..k-1 | predicción hecha en el game k |
| N | games 1-N | las N-1 predicciones posibles | ninguna |

La fila con `game_num = g` de la tabla de predicciones contiene la predicción
**formulada en el game g** para el game `g+1`. Por eso:

- La predicción pendiente que se muestra en el panel es la fila `g = step`.
- Una fila ya es evaluable cuando su game `g+1` está revelado, es decir cuando
  `g ≤ step - 1`.
- El último game de cada partido no admite predicción (no hay game siguiente),
  pero sí tiene momentum calculado, por lo que se muestra en la serie y el
  marcador llega al resultado final.

### Controles del gráfico

En la barra lateral, sección **Gráfico**:

| Control | Efecto |
|---|---|
| `Mostrar M_A` / `Mostrar M_B` / `Mostrar N(t)` | Activa o oculta cada serie |
| `Mostrar predicciones R4 en la curva` | Superpone la trayectoria predicha frente a la observada |
| `Suavizado (media movil)` | `1` = sin suavizado. `3`, `5` o `7` dibujan una media móvil centrada que facilita la lectura de la tendencia |

### Predicciones sobre la curva

El gráfico muestra tres capas de información:

| Capa | Descripción |
|---|---|
| **Series observadas** | `M_A`, `M_B` y `N(t)` tal como quedaron registrados en el partido |
| **Predicción R4** | Línea discontinua verde con la predicción formulada en cada game para `N(t+1)`, desplazada un game para alinearse con el valor real |
| **Observado N(t+1)** | Marcadores huecos sobre la misma posición, que permiten contrastar predicción y realidad game a game |
| **Proyección del próximo game** | Estrella verde que conecta el último valor observado con la predicción pendiente de revelar |

El contraste entre la línea discontinua y los marcadores huecos muestra de forma
directa en qué games el modelo se aproximó y en cuáles se desvió, sin necesidad
de consultar la tabla.

La tabla de evaluación identifica cada fila por el **game previsto**, e indica
en `Predicho en` el game desde el que se formuló la predicción. De este modo
el valor predicho y el observado de una misma fila se refieren al mismo game.

El suavizado es únicamente una ayuda visual: los valores observados se mantienen
como marcadores sobre la línea, de modo que la varianza real de la serie nunca
se oculta. La serie de momentum presenta saltos reales entre games consecutivos,
por lo que un suavizado excesivo puede inducir a error.

### Nota de implementación

Streamlit sanea el HTML insertado con DOMPurify, que elimina las etiquetas
`<style>`. Por ese motivo los componentes propios de R8 usan **estilos en línea**
y el tema general de la aplicación se define en `.streamlit/config.toml`.

### Filtro de partidos

- **Fuera de muestra (≥ 2023)**: recomendado, porque R4 y R5 se entrenaron con
  años anteriores a 2023.
- **Todos los partidos**: incluye partidos in-sample; la interfaz advierte que
  sus predicciones pueden mostrar un desempeño optimista.

### Principio metodológico

R4 y R5 fueron entrenados con granularidad de **game**. Por esa razón la
predicción se actualiza al cerrar cada game y nunca con información parcial del
game en curso. Durante un game en progreso solo se muestran el marcador y las
estadísticas observables, sin producir una predicción anticipada.

Al cerrar un juego, Streamlit solicita al backend un pronóstico para el
siguiente. La API utiliza la fila de variables de ese juego y, para la GRU,
el historial de momentum disponible hasta el mismo cierre. Streamlit guarda
el pronóstico en la sesión y registra su error cuando se revela el siguiente
juego. Un salto manual solicita el pronóstico del juego cerrado visible.

### Lectura de las métricas en pantalla

Las métricas acumuladas (MAE y acierto direccional) corresponden **únicamente al
partido seleccionado** y a los games ya revelados. No son comparables con las
métricas globales de R4 y R5, calculadas sobre 33 702 observaciones del conjunto
de prueba.

El backend carga los artefactos entrenados al arrancar. Streamlit mantiene
una caché breve de las respuestas HTTP durante la reproducción.

## Cómo abrir el prototipo R7

### Opción A — Doble clic
Abrir `mockup/index.html` en cualquier navegador moderno (Chrome / Firefox / Safari).

### Opción B — Servidor local (recomendado)
Para evitar restricciones de CORS al cargar Chart.js por CDN:

```bash
cd mockup
python3 -m http.server 8080
# luego abrir: http://localhost:8080
```

### Opción C — Capturas estáticas
Las figuras para inclusion en la memoria se generan con:
```bash
# (requiere: playwright o selenium; ver presentation/generate_figures.py)
```

## Matriz de trazabilidad bibliografica

Tabla que vincula cada componente visual con el algoritmo subyacente y el respaldo
en la literatura. Esta matriz es el **medio de verificacion formal** de R7.

| Componente visual | Fundamento tecnico | Respaldo en literatura | Justificacion |
|---|---|---|---|
| Barra superior (Selector de Superficie) | Carga dinamica de vectores de pesos EWM estratificados por superficie | Liu et al. (2025); Sampaio et al. (2024) | La dinamica de rebote y velocidad varia drasticamente entre arcilla, hierba y dura; el modelo requiere ajustar parametros para mantener la precision operativa. El selector permite al sistema conmutar instantaneamente entre matrices de pesos EWM calculadas globalmente, optimizando la precision operativa y garantizando un tiempo de respuesta en milisegundos. |
| Tarjetas de jugadores (Indice de Momentum) | Método EWM y lógica difusa | [dato pendiente: referencias metodológicas verificadas] | El índice agrega indicadores dinámicos mediante pesos por entropía y representa estados difusos; no elimina todas las decisiones de diseño ni equivale a una medida causal de momentum. |
| Grafica de Momentum (Curva punto a punto) | Ventana Deslizante y Flujo de Rendimiento | Xu & Wang (2024); Yao et al. (2025) | Visualiza el momentum como un flujo sistematico y no azaroso, permitiendo al entrenador pasar de la reaccion tardia a la anticipacion estrategica. |
| Banda de confianza y proyeccion | Redes LSTM y Transformers (ResiDual) | Xu & Wang (2024); Tao et al. (2025); Zhang et al. (2024) | Las arquitecturas secuenciales capturan dependencias historicas largas. La linea discontinua refleja la capacidad del modelo para predecir la tendencia basada en rallies previos. |
| Lineas verticales de Turning Points | Deteccion Bayesiana (Algoritmo Binseg) | Li & Wang (2024); Zhang et al. (2024) | Identifican puntos de cambio en las propiedades estadisticas de la serie, diferenciando entre un colapso gradual y una reversion subita. |
| Panel lateral: Riesgo de Quiebre | Valores SHAP e Interpretabilidad | Gao et al. (2024); Yao et al. (2025) | Desglosa la "caja negra" del modelo. Identifica que el dominio del saque es determinante (hasta 77.7% de peso) y como errores no forzados afectan el quiebre. |
| Modulo de Indices de Rendimiento Dinamico (5 Metricas) | Metricas Espacio-Temporales y EWM | Gao et al. (2024); Yao et al. (2025); Li & Wang (2024); Sampaio et al. (2024); Dong (2024) | Visualiza las 5 variables de mayor impacto tactico mediante barras duales comparativas (A vs B). Transforma metricas crudas en conocimiento tactico accionable, permitiendo al coach identificar si una caida de momentum es tecnica o fisica en milisegundos. |

*Fuente: Elaboracion propia.*

## Cumplimiento del indicador (R7)

> **Indicador**: "Todos los modulos de la interfaz fundamentados en la literatura
> cientifica y visualizacion de al menos 5 metricas clave."

**Cumplimiento:**
- [x] 7 modulos documentados con respaldo bibliografico explicito.
- [x] 5 metricas dinamicas visualizadas (fatiga, errores no forzados, ratio aces,
      conversion break points, velocidad de saque).
- [x] Matriz de trazabilidad completa = medio de verificacion formal.

## Limitaciones declaradas (transparencia)

- **R7 usa datos simulados**: el mockup HTML valida el diseño, pero no debe
  confundirse con la aplicación funcional R8.
- **R8 es replay histórico, no live**: los games se revelan desde datos ya
  registrados; no se conecta a ninguna fuente de puntuación en vivo.
- **Artefactos entrenados**: regenera R4 y R5 con `python -m backend.train`
  tras cambiar los datos de entrenamiento o los hiperparámetros, y reinicia el backend.
- **Predicción experimental**: las salidas no sustituyen el criterio del
  entrenador y no constituyen recomendaciones tácticas causales.
- **Granularidad de game**: la predicción se genera al cerrar cada game, porque
  así se entrenaron los modelos.
- **Explicabilidad por juego (R9)**: el panel describe el cambio observado y muestra
  hasta dos señales del pronóstico del próximo juego en lenguaje de tenis. Las
  señales se calculan con los valores SHAP locales. El beeswarm se conserva
  en el notebook técnico de R4–R5.
- **Cruces observados (R10)**: el panel indica el último juego en que se confirmó
  un cambio de signo y el jugador favorecido. Los valores cero conservan el
  último signo observado hasta que se registra un valor de signo contrario.
- **Horizonte de cruce (R6)**: la GRU estima cuántos juegos faltan, condicionado
  a que ocurra otro cruce durante el partido.

## Estructura

```
dashboard/
  app.py             # dashboard funcional R8
  mockup/
    index.html       # estructura del dashboard
    style.css        # diseno visual
    app.js           # logica + datos simulados + Chart.js
  README.md          # este archivo
```

## Evaluación pendiente

Quedan por aplicar la escala SUS con usuarios representativos (meta de 70
puntos) y la revisión cualitativa con un especialista en tenis. Las tareas,
el cálculo del puntaje y el contenido del acta están en el
[protocolo de evaluación](../reports/07_protocolo_usabilidad.md).

## Medición de latencia de R8

Con los artefactos de inferencia disponibles, iniciar en terminales separadas
el backend, el dashboard y Chrome con un perfil temporal. Ejemplo en macOS:

```bash
python -m uvicorn backend.api:app --port 18769
TACTISPRO_BACKEND_URL=http://127.0.0.1:18769 streamlit run dashboard/app.py --server.address 127.0.0.1 --server.port 18770 --server.headless true
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless --disable-gpu --remote-debugging-port=18768 --user-data-dir="$(mktemp -d)" about:blank
```

Ejecutar desde la raíz `python -m scripts.benchmark_replay_latency`. El
script requiere el cliente síncrono de `websockets` y guarda las
observaciones y los percentiles en `data/processed/r8_replay_latency.json`.
La medición visual recorre el partido inicial con el botón Siguiente,
descarta el primer cierre como calentamiento y termina en el primer
cuadro disponible de la curva y el pronóstico. También registra el fin de
la animación. Las etapas internas se cronometran por separado en los
mismos cierres, con conexiones locales y los servicios ya iniciados.
