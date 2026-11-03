"""Interfaz para revisar reseñas de huéspedes y cliente de la API."""

import csv
import io
import os
from hashlib import sha256
from itertools import batched

import pandas as pd
import requests
import streamlit as st


API_URL = os.environ.get("RESENAS_API_URL", "http://127.0.0.1:8000").rstrip("/")
MAX_CARACTERES = 10_000
MAX_LOTE = 500
ETIQUETAS = ("positiva", "negativa")


class ErrorAnalisis(Exception):
    """Problema que la interfaz puede mostrar al administrador."""


def solicitar(ruta, datos):
    try:
        respuesta = requests.post(f"{API_URL}{ruta}", json=datos, timeout=(3, 30))
        respuesta.raise_for_status()
        return respuesta.json()
    except requests.Timeout as error:
        raise ErrorAnalisis("El análisis tardó demasiado. Intenta nuevamente.") from error
    except requests.ConnectionError as error:
        raise ErrorAnalisis("El servicio de análisis no está disponible. Intenta nuevamente.") from error
    except requests.RequestException as error:
        raise ErrorAnalisis("No pudimos completar el análisis. Intenta nuevamente.") from error


def sentimiento(resultado):
    etiqueta = resultado.get("sentimiento") if isinstance(resultado, dict) else None
    if etiqueta not in ETIQUETAS:
        raise ErrorAnalisis("Recibimos una respuesta inesperada. Intenta nuevamente.")
    return etiqueta


def analizar_resena(texto):
    return sentimiento(solicitar("/predecir", {"texto": texto}))


def analizar_lote(textos):
    resultados = []
    for bloque in batched(textos, MAX_LOTE):
        respuesta = solicitar("/predecir/lote", {"resenas": [{"texto": texto} for texto in bloque]})
        lote = respuesta.get("resultados") if isinstance(respuesta, dict) else None
        if not isinstance(lote, list) or len(lote) != len(bloque):
            raise ErrorAnalisis("El análisis no devolvió un resultado por reseña. Intenta nuevamente.")
        resultados.extend(sentimiento(elemento) for elemento in lote)
    return resultados


def leer_csv(datos):
    try:
        texto = datos.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ErrorAnalisis("Guarda el archivo con codificación UTF-8 y vuelve a cargarlo.") from error
    try:
        dialecto = csv.Sniffer().sniff(texto[:65_536], delimiters=",;\t")
        separador = dialecto.delimiter
    except csv.Error:
        # También se admiten CSV de una única columna sin separadores.
        separador = ","
    lector = csv.reader(io.StringIO(texto, newline=""), delimiter=separador, strict=True)
    try:
        cabecera = next(lector, [])
        columnas = [columna.strip() for columna in cabecera]
        if not columnas or any(not columna for columna in columnas):
            raise ErrorAnalisis("El CSV necesita una primera fila con los nombres de las columnas.")
        if len({columna.casefold() for columna in columnas}) != len(columnas):
            raise ErrorAnalisis("Cada columna del CSV necesita un nombre distinto.")
        filas = []
        for fila in lector:
            if not fila:
                continue
            if len(fila) != len(columnas):
                raise ErrorAnalisis(f"Revisa el número de columnas cerca de la línea {lector.line_num}.")
            filas.append(fila)
    except csv.Error as error:
        raise ErrorAnalisis(f"No pudimos leer el CSV cerca de la línea {lector.line_num}. Revisa su formato.") from error
    if not filas:
        raise ErrorAnalisis("Agrega al menos una reseña debajo del encabezado.")
    return columnas, filas


def exportar_csv(columnas, filas):
    def celda_literal(valor):
        # Evita que una hoja de cálculo ejecute expresiones aportadas en el CSV.
        if valor.startswith(("\t", "\r", "\n")) or valor.lstrip().startswith(("=", "+", "-", "@")):
            return "'" + valor
        return valor

    salida = io.StringIO(newline="")
    escritor = csv.writer(salida)
    escritor.writerow([celda_literal(columna) for columna in columnas])
    escritor.writerows([celda_literal(valor) for valor in fila] for fila in filas)
    return salida.getvalue().encode("utf-8-sig")


def mostrar_etiqueta(etiqueta):
    mensaje = f"Clasificación sugerida: {etiqueta}"
    if etiqueta == "negativa":
        st.warning(mensaje, icon=":material/flag:")
    else:
        st.success(mensaje, icon=":material/check_circle:")


def reseña_individual():
    with st.form("analizar_individual", border=False):
        texto = st.text_area(
            "Texto de la reseña", height=185, max_chars=MAX_CARACTERES,
            placeholder="Escribe o pega el comentario del huésped…",
        )
        enviado = st.form_submit_button("Analizar reseña", type="primary", icon=":material/arrow_forward:")
    if enviado:
        st.session_state.pop("resultado_individual", None)
        if not texto.strip():
            st.warning("Escribe una reseña para analizarla.")
        else:
            try:
                with st.spinner("Analizando la reseña…"):
                    st.session_state["resultado_individual"] = analizar_resena(texto.strip())
            except ErrorAnalisis as error:
                st.error(str(error))
    if etiqueta := st.session_state.get("resultado_individual"):
        mostrar_etiqueta(etiqueta)


def cargar_archivo():
    archivo = st.file_uploader(
        "Archivo de reseñas", type=["csv"], max_upload_size=10,
        help="Usa CSV en UTF-8 con encabezados. Admite coma, punto y coma o tabulación.",
    )
    if archivo is None:
        st.session_state.pop("resultado_csv", None)
        return
    datos = archivo.getvalue()
    try:
        columnas, filas = leer_csv(datos)
    except ErrorAnalisis as error:
        st.session_state.pop("resultado_csv", None)
        st.error(str(error))
        return

    nombres_comunes = {"texto", "reseña", "resena", "review_text", "comentario"}
    sugerida = next((i for i, nombre in enumerate(columnas) if nombre.casefold() in nombres_comunes), 0)
    huella = sha256(datos).hexdigest()
    columna = st.selectbox("Columna con el texto de las reseñas", columnas, index=sugerida, key=f"columna_{huella}")
    indice = columnas.index(columna)
    clave = (huella, columna)
    if st.button("Analizar archivo", type="primary", icon=":material/arrow_forward:", key=f"analizar_csv_{huella}"):
        st.session_state.pop("resultado_csv", None)
        st.session_state.pop("filtro_csv", None)
        textos = [fila[indice].strip() for fila in filas]
        invalidas = [str(i) for i, texto in enumerate(textos, start=1) if not 1 <= len(texto) <= MAX_CARACTERES]
        if invalidas:
            lista = ", ".join(invalidas[:8]) + (", …" if len(invalidas) > 8 else "")
            st.error(f"Revisa las reseñas {lista}. Cada texto debe tener entre 1 y 10.000 caracteres.")
        else:
            try:
                with st.spinner("Analizando las reseñas…"):
                    st.session_state["resultado_csv"] = clave, analizar_lote(textos)
            except ErrorAnalisis as error:
                st.error(str(error))

    resultado = st.session_state.get("resultado_csv")
    if resultado is not None and resultado[0] == clave:
        mostrar_resultados(columnas, filas, indice, resultado[1])


def mostrar_resultados(columnas, filas, indice, etiquetas):
    st.divider()
    st.subheader("Resultados")
    filtros = {"Todas": None, "Negativas": "negativa", "Positivas": "positiva"}
    opcion = st.radio("Filtrar por clasificación", list(filtros), horizontal=True, key="filtro_csv")
    visibles = [
        (numero, fila, etiqueta) for numero, (fila, etiqueta) in enumerate(zip(filas, etiquetas, strict=True), 1)
        if filtros[opcion] is None or filtros[opcion] == etiqueta
    ]
    if not visibles:
        st.info("No hay reseñas para este filtro.")
        return

    tabla = pd.DataFrame(
        [(numero, etiqueta.capitalize(), fila[indice]) for numero, fila, etiqueta in visibles],
        columns=["Reseña", "Clasificación", "Texto"],
    )
    st.dataframe(tabla, width="stretch", hide_index=True,
                 column_config={"Texto": st.column_config.TextColumn(width="large")})
    with st.expander("Leer una reseña completa"):
        detalles = {numero: (fila, etiqueta) for numero, fila, etiqueta in visibles}
        elegida = st.selectbox("Número de reseña", list(detalles), key=f"detalle_csv_{opcion}")
        fila, etiqueta = detalles[elegida]
        mostrar_etiqueta(etiqueta)
        st.text(fila[indice])

    nombres = {nombre.casefold() for nombre in columnas}
    nueva_columna = "sentimiento"
    while nueva_columna.casefold() in nombres:
        nueva_columna += "_analizado"
    accion = "Descargar todas las reseñas" if opcion == "Todas" else f"Descargar reseñas {opcion.lower()}"
    st.download_button(
        accion, data=exportar_csv(
            [*columnas, nueva_columna], ([*fila, etiqueta] for _, fila, etiqueta in visibles)
        ),
        file_name=f"resenas_{opcion.lower()}.csv", mime="text/csv", icon=":material/download:",
        key=f"descargar_csv_{opcion}", on_click="ignore",
    )


st.set_page_config(page_title="Reseñas del hotel", page_icon=":material/rate_review:", layout="centered")
st.title("Reseñas del hotel")
st.caption("Revisa comentarios de huéspedes y encuentra los que requieren atención.")
individual, archivo_csv = st.tabs(["Una reseña", "Archivo CSV"])
with individual:
    reseña_individual()
with archivo_csv:
    cargar_archivo()
