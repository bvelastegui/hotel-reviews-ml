# Análisis de reseñas hoteleras

Aplicación en Python para ayudar al personal de un hotel a revisar comentarios de huéspedes. Permite analizar una reseña o un archivo CSV, filtrar las sugerencias por sentimiento y descargar las reseñas visibles.

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

Los lotes admiten hasta 500 textos y mantienen el orden de entrada. Cada reseña admite hasta 10.000 caracteres. Las entradas inválidas reciben HTTP 422.

## Analizar reseñas

- En **Una reseña**, pegar un comentario y pulsar **Analizar reseña**.
- En **Archivo CSV**, cargar el [archivo de ejemplo](ejemplos/resenas_demo.csv), seleccionar la columna `texto` y pulsar **Analizar archivo**. Se aceptan CSV UTF-8 con encabezados de hasta 10 MB, separados por coma, punto y coma o tabulación.
- En **Resultados**, filtrar por sentimiento, abrir una reseña completa y descargar las filas visibles con su clasificación. Las clasificaciones son sugerencias para la revisión humana.
