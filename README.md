# Análisis de reseñas hoteleras

**Integrantes:** Bryan Velastegui y Evelyn Morocho.

Aplicación en Python para ayudar al personal de un hotel a revisar comentarios de huéspedes. Permite analizar una reseña o un archivo CSV, filtrar las sugerencias por sentimiento, agrupar quejas parecidas y descargar las reseñas visibles para apoyar decisiones.

## Preparar el modelo

Usar Python 3.12 o superior. Desde la carpeta del proyecto, con un entorno virtual activo, ejecutar:

```bash
python -m pip install -r requirements.txt
jupyter lab modelo_resenas.ipynb
```

Abrir el [cuaderno](modelo_resenas.ipynb) y ejecutar las celdas en orden. El cliente oficial de Kaggle descarga el corpus con las credenciales locales cuando hace falta. El cuaderno verifica el archivo, prepara las particiones, compara clasificadores y guarda el pipeline junto con su registro de métricas en `resultados/modelo/`.

En la primera celda de código se pueden configurar la semilla y los hiperparámetros. Al explorar cambios, usar `EVALUAR_PRUEBA = False` para reservar la prueba final. El cuaderno incluye los gráficos de distribución, validación, matriz de confusión y desempeño por pliegue.

El corpus [Andalusian Hotels’ Reviews v3](https://www.kaggle.com/datasets/chizhikchi/andalusian-hotels-reviews-unbalanced/versions/3), publicado por Mariia Chizhikova, declara licencia [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Las etiquetas se derivan de las estrellas. La preparación excluye tres estrellas y agrupa coincidencias de texto antes de dividir los datos. El modelo seleccionado, TF-IDF con Complement Naive Bayes, obtuvo un F1 macro de 0,9183 en la prueba reservada.

## Ejecutar la aplicación

Con el pipeline ya generado, iniciar la API y la interfaz desde la carpeta del proyecto en terminales separadas:

```bash
python -m uvicorn backend:app --host 127.0.0.1 --port 8000
```

```bash
python -m streamlit run frontend.py --server.address 127.0.0.1 --server.port 8501
```

Abrir la [interfaz](http://127.0.0.1:8501). Si la API utiliza otro puerto, configurar `RESENAS_API_URL` antes de iniciar Streamlit. La [documentación interactiva de la API](http://127.0.0.1:8000/docs) está disponible con el servidor en ejecución.

| Operación | Entrada | Salida |
| --- | --- | --- |
| `GET /salud` | Sin cuerpo | `{"estado":"listo"}` |
| `POST /predecir` | `{"texto":"La habitación estaba impecable."}` | `{"sentimiento":"positiva"}` |
| `POST /predecir/lote` | `{"resenas":[{"texto":"La habitación estaba impecable."}]}` | `{"resultados":[{"sentimiento":"positiva"}]}` |
| `POST /agrupar/negativas` | `{"resenas":[{"numero":2,"texto":"La habitación estaba sucia."}]}` | Total y grupos con identificador, palabras, cantidad y números de reseña |

Los lotes admiten hasta 500 textos y mantienen el orden de entrada. Cada reseña admite hasta 10.000 caracteres. Las entradas inválidas reciben HTTP 422.

## Analizar reseñas

- En **Una reseña**, pegar un comentario y pulsar **Analizar reseña**.
- En **Archivo CSV**, cargar el [archivo de ejemplo](ejemplos/resenas_demo.csv), seleccionar la columna `texto` y pulsar **Analizar archivo**. Se aceptan CSV UTF-8 con encabezados de hasta 10 MB, separados por coma, punto y coma o tabulación.
- En **Resultados**, filtrar por sentimiento, abrir una reseña completa y descargar las filas visibles con su clasificación. Las clasificaciones son sugerencias para la revisión humana.
- En **Problemas que se repiten**, pulsar **Agrupar reseñas negativas**. Revisar los grupos ordenados por cantidad, seleccionar uno, leer sus comentarios y usar **Descargar quejas agrupadas** para exportar las filas visibles con su grupo. Se conservan las columnas originales y se añaden nombres distintos si ya existen columnas de resultado.

La agrupación admite hasta 5.000 reseñas negativas por archivo y diez millones de caracteres en total, con números de reseña únicos. El número corresponde a la posición de la reseña en el CSV, empezando en uno. Las reseñas repetidas se cuentan como filas independientes y los textos sin vocabulario utilizable se conservan en un grupo aparte. Un lote vacío devuelve cero grupos.

Los grupos se calculan sobre el archivo recibido mediante TF-IDF y K-means; cada comentario se asigna a uno y los tres términos con mayor peso medio sirven como nombre. La cantidad es exploratoria, hasta cinco grupos con vocabulario y un posible grupo sin términos. Los identificadores solo tienen sentido dentro de ese archivo. La frecuencia ayuda a decidir qué revisar primero, pero no establece la gravedad ni garantiza que todos los comentarios compartan el mismo problema. Conviene leer las reseñas antes de actuar; una reseña puede hablar de varios asuntos y la clasificación inicial puede equivocarse.

La sección 9 del cuaderno reproduce la misma función del backend sobre las negativas de entrenamiento y guarda un gráfico en `resultados/figuras_modelo/05_grupos_de_quejas.png`. El registro `resultados/agrupacion_exploratoria.json` conserva la correspondencia con los identificadores del corpus. Esta exploración no utiliza la prueba reservada ni cambia el clasificador de sentimiento. Referencia: [agrupación de documentos con K-means](https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html).
