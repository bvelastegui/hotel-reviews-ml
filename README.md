# Análisis de reseñas hoteleras

**Integrantes:** Bryan Velastegui, Evelyn Morocho y Erika Gudiño.

Aplicación en Python para ayudar al personal de un hotel a revisar comentarios de huéspedes. Permite analizar una reseña o un archivo CSV, filtrar las sugerencias por sentimiento, agrupar quejas parecidas y descargar las reseñas visibles para apoyar decisiones.

## Preparar el modelo

Usar Python 3.12 o superior. Desde la carpeta del proyecto, con un entorno virtual activo, ejecutar:

```bash
python -m pip install -r requirements.txt
jupyter lab modelo_resenas.ipynb
```

Abrir el [cuaderno](modelo_resenas.ipynb) y ejecutar las celdas en orden. El cliente oficial de Kaggle descarga el corpus con las credenciales locales cuando hace falta. El cuaderno verifica el archivo, prepara las particiones, compara clasificadores, ajusta el umbral de decisión en validación y guarda el pipeline de referencia junto con su registro de métricas en `resultados/modelo/`. **Todo el entrenamiento y la creación del modelo ocurren en el cuaderno**; ningún otro archivo entrena.

La celda de parámetros (etiquetada `parameters`) concentra las familias de modelos, las rejillas de hiperparámetros, la representación TF-IDF, la meta de recall de negativas, la validación cruzada y la carpeta de salida. Al explorar cambios, usar `EVALUAR_PRUEBA = False` para reservar la prueba final. El cuaderno incluye los gráficos de distribución, comparación de candidatos, umbral, matriz de confusión, validación cruzada, método del codo y grupos de quejas.

El corpus [Andalusian Hotels’ Reviews v3](https://www.kaggle.com/datasets/chizhikchi/andalusian-hotels-reviews-unbalanced/versions/3), publicado por Mariia Chizhikova, declara licencia [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Las etiquetas se derivan de las estrellas. La preparación excluye tres estrellas y agrupa coincidencias de texto antes de dividir los datos. Se compararon 22 configuraciones (SVM lineal, Complement Naive Bayes, regresión logística, KNN, árbol de decisión y una referencia mayoritaria). El modelo seleccionado en validación, TF-IDF con SVM lineal (C = 2, pesos balanceados) y umbral ajustado para detectar al menos el 90 % de las negativas de validación, obtuvo en la prueba reservada F1 macro de 0,9350 y detectó 373 de 410 reseñas negativas.

## Ejecutar la aplicación

Con el pipeline ya generado, iniciar todo desde la carpeta del proyecto con un solo comando:

```bash
python iniciar.py
```

El script comprueba el modelo, inicia la API, espera a que responda y abre la [aplicación del hotel](http://127.0.0.1:8501) y la [herramienta de entrenamiento](http://127.0.0.1:8502). Ctrl+C detiene los tres servicios. `python iniciar.py --help` muestra cómo cambiar los puertos.

También se pueden iniciar por separado:

```bash
python -m uvicorn backend:app --host 127.0.0.1 --port 8000
python -m streamlit run frontend.py --server.address 127.0.0.1 --server.port 8501
python -m streamlit run entrenamiento.py --server.address 127.0.0.1 --server.port 8502
```

Si la API utiliza otro puerto, configurar `RESENAS_API_URL` antes de iniciar Streamlit. La [documentación interactiva de la API](http://127.0.0.1:8000/docs) está disponible con el servidor en ejecución.

| Operación | Entrada | Salida |
| --- | --- | --- |
| `GET /salud` | Sin cuerpo | `{"estado":"listo"}` |
| `POST /predecir` | `{"texto":"La habitación estaba impecable."}` | `{"sentimiento":"positiva"}` |
| `POST /predecir/lote` | `{"resenas":[{"texto":"La habitación estaba impecable."}]}` | `{"resultados":[{"sentimiento":"positiva"}]}` |
| `POST /agrupar/negativas` | `{"resenas":[{"numero":2,"texto":"La habitación estaba sucia."}]}` | Total y grupos con identificador, palabras, cantidad y números de reseña |
| `GET /modelo` | Sin cuerpo | Algoritmo, parámetros, umbral y métricas del modelo en uso |
| `POST /modelo/activar` | `{"id":"20261119-150108-e58d5c"}` | Pone en uso el pipeline de esa ejecución del cuaderno, tras verificar su huella |
| `POST /modelo/restaurar` | Sin cuerpo | Vuelve al modelo de referencia del cuaderno |

Las cuatro primeras operaciones atienden a la aplicación del hotel; las tres últimas, a la herramienta de entrenamiento. La API no entrena: solo carga, describe, activa o restaura pipelines producidos por el cuaderno. Los lotes admiten hasta 500 textos y mantienen el orden de entrada. Cada reseña admite hasta 10.000 caracteres. Las entradas inválidas reciben HTTP 422.

## Entrenar y comparar modelos

La [herramienta de entrenamiento](entrenamiento.py) es para el equipo técnico y está separada de la aplicación del hotel, que no muestra métricas ni parámetros. No contiene código de entrenamiento: traduce el formulario a los parámetros del cuaderno y lo ejecuta con [papermill](https://papermill.readthedocs.io/).

- Elegir el algoritmo (SVM lineal, Complement Naive Bayes, regresión logística, KNN, árbol de decisión o referencia mayoritaria) y sus parámetros: C, pesos de clase, alpha, vecinos o profundidad.
- Ajustar la representación TF-IDF (n-gramas, frecuencia mínima y máximo de términos), la meta de recall de negativas para el umbral y, opcionalmente, la validación cruzada de 5 pliegues por grupos.
- Pulsar **Entrenar**. Se ejecuta `modelo_resenas.ipynb` con esos parámetros (`EVALUAR_PRUEBA = False`, sin exploración de grupos) y salida en `resultados/modelos/<id>/`, que guarda el cuaderno ejecutado, el pipeline y sus métricas. La herramienta muestra las métricas de validación que registró el cuaderno: F1 macro, exactitud, precisión y recall por clase, matriz de confusión y F1 por pliegue, comparadas con el modelo en uso. La prueba reservada nunca se consulta. El cuaderno ejecutado se puede descargar como evidencia.
- **Usar este modelo en la aplicación** pide a la API que active esa ejecución (`POST /modelo/activar`), sin reiniciarla; la selección se conserva al reiniciar. **Restaurar el modelo de referencia** vuelve al pipeline de `resultados/modelo/`.

## Archivos de Python y su justificación

| Archivo | Por qué existe |
| --- | --- |
| `modelo_resenas.ipynb` | Único lugar donde se adquieren y limpian los datos, se entrena, se ajusta el umbral, se evalúa y se exporta el modelo. |
| `backend.py` | API exigida por el enunciado: clasifica casos nuevos, agrupa quejas y carga, activa o restaura pipelines del cuaderno verificando su huella. No entrena. |
| `frontend.py` | Interfaz gráfica del administrador: análisis, filtros, búsqueda, grupos y descargas. |
| `entrenamiento.py` | Interfaz técnica que cubre el requisito de entrenar, configurar parámetros y ver métricas desde la aplicación, ejecutando el cuaderno con papermill. |
| `iniciar.py` | Arranque de la demostración: inicia la API, espera a que responda, abre las dos interfaces y las detiene juntas. No contiene lógica del modelo. |

## Analizar reseñas

La aplicación comparte paleta y tipografía (Inter, alojada en `static/fuentes/` con su licencia OFL) con la presentación y el póster, para que funcione igual sin conexión.


- La vista principal está pensada para el uso habitual: cargar un CSV de reseñas, como el [ejemplo extendido](ejemplos/resenas_demo_extendido.csv) (60 reseñas ficticias con `id`, `fecha`, `canal` y `texto`, que incluyen quejas recurrentes, reseñas mixtas, ironía, negaciones y textos cortos) o el [ejemplo corto](ejemplos/resenas_demo.csv), seleccionar la columna `texto` y pulsar **Analizar archivo**. La aplicación propone la columna que más se parece a texto libre y muestra un ejemplo de su contenido. Si se elige una columna que no parece contener reseñas (menos del 60 % de sus filas tiene al menos tres palabras, como ocurre con fechas, identificadores o categorías), explica qué contiene, sugiere otra columna y no permite analizarla. Se aceptan CSV UTF-8 con encabezados de hasta 10 MB, separados por coma, punto y coma o tabulación.
- Para un comentario suelto, abrir **Revisar una reseña** en la cabecera, pegar el texto y pulsar **Analizar reseña**.
- En **Resultados**, un resumen indica cuántas reseñas del archivo son negativas. Filtrar por clasificación (cada opción muestra su cantidad), buscar una palabra en el texto sin importar mayúsculas ni tildes, seleccionar una fila para leer la reseña completa y descargar las filas visibles con su clasificación. La tabla muestra también hasta dos columnas breves del propio archivo, como un identificador o una fecha; son datos del usuario, no resultados del análisis. Las clasificaciones son sugerencias para la revisión humana.
- Sin archivo cargado, la vista explica los tres pasos y **Descargar plantilla** ofrece un CSV de ejemplo con el formato esperado.
- En **Problemas que se repiten**, pulsar **Agrupar reseñas negativas**. Los grupos aparecen como botones ordenados por cantidad: pulsar uno muestra solo sus quejas y pulsarlo de nuevo muestra todas. Seleccionar una fila para leer la queja completa y usar **Descargar** para exportar las filas visibles con su grupo. Se conservan las columnas originales y se añaden nombres distintos si ya existen columnas de resultado.

La agrupación admite hasta 5.000 reseñas negativas por archivo y diez millones de caracteres en total, con números de reseña únicos. El número corresponde a la posición de la reseña en el CSV, empezando en uno. Las reseñas repetidas se cuentan como filas independientes y los textos sin vocabulario utilizable se conservan en un grupo aparte. Un lote vacío devuelve cero grupos.

Los grupos se calculan sobre el archivo recibido mediante TF-IDF y K-means; cada comentario se asigna a uno y los tres términos con mayor peso medio sirven como nombre. La cantidad es exploratoria, hasta cinco grupos con vocabulario y un posible grupo sin términos. Los identificadores solo tienen sentido dentro de ese archivo. La frecuencia ayuda a decidir qué revisar primero, pero no establece la gravedad ni garantiza que todos los comentarios compartan el mismo problema. Conviene leer las reseñas antes de actuar; una reseña puede hablar de varios asuntos y la clasificación inicial puede equivocarse.

La sección 10 del cuaderno aplica el método del codo y el coeficiente de silueta para elegir la cantidad de grupos: no aparece un codo claro y la silueta queda cerca de cero, porque las quejas comparten vocabulario, así que se mantiene un máximo de cinco grupos por criterio de uso. Esa sección reproduce además la misma función del backend sobre las negativas de entrenamiento y guarda los gráficos `06_codo_y_silueta.png` y `07_grupos_de_quejas.png` en `resultados/figuras_modelo/`. El registro `resultados/agrupacion_exploratoria.json` conserva la correspondencia con los identificadores del corpus. Esta exploración no utiliza la prueba reservada ni cambia el clasificador de sentimiento. Referencia: [agrupación de documentos con K-means](https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html).

## Informe

El [informe final en PDF](entregables/informe.pdf) documenta los datos, la metodología, los resultados y el uso de la aplicación.

## Presentación y póster

La [presentación](entregables/presentacion.pptx) incluye diapositivas principales y de respaldo, con notas del expositor. También se incluye una [copia PDF](entregables/presentacion.pdf).

El [póster en PDF](entregables/poster.pdf) tiene formato A3 horizontal (420 × 297 mm).
