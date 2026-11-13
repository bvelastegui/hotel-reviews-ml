"""Interfaz para revisar reseñas de huéspedes y cliente de la API."""

import csv
import io
import os
import unicodedata
from hashlib import sha256
from itertools import batched
from pathlib import Path

import pandas as pd
import requests
import streamlit as st


MARCA = Path(__file__).resolve().parent / "static" / "marca.svg"
API_URL = os.environ.get("RESENAS_API_URL", "http://127.0.0.1:8000").rstrip("/")
MAX_CARACTERES = 10_000
MAX_LOTE = 500
MAX_AGRUPACION = 5_000
ETIQUETAS = ("positiva", "negativa")
PLANTILLA = "id,texto\nR-001,La habitación estaba limpia y el personal fue muy amable.\nR-002,El aire acondicionado no funcionaba y nadie vino a revisarlo.\n"


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


def agrupar_quejas(negativas, indice):
    respuesta = solicitar("/agrupar/negativas", {"resenas": [
        {"numero": numero, "texto": fila[indice].strip()} for numero, fila in negativas
    ]})
    grupos = respuesta.get("grupos") if isinstance(respuesta, dict) else None
    valido = isinstance(grupos, list) and respuesta.get("total") == len(negativas)
    numeros, identificadores = [], []
    if valido:
        for grupo in grupos:
            if (not isinstance(grupo, dict) or type(grupo.get("id")) is not int
                    or not isinstance(grupo.get("tema"), str) or not grupo["tema"].strip()
                    or not isinstance(grupo.get("resenas"), list)
                    or not grupo["resenas"] or grupo.get("cantidad") != len(grupo["resenas"])
                    or any(type(numero) is not int for numero in grupo["resenas"])):
                valido = False
                break
            identificadores.append(grupo["id"])
            numeros.extend(grupo["resenas"])
    if (not valido or len(set(identificadores)) != len(identificadores)
            or sorted(numeros) != sorted(numero for numero, _ in negativas)):
        raise ErrorAnalisis("La agrupación no devolvió todas las quejas. Intenta nuevamente.")
    return grupos


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


def nombre_disponible(nombre, columnas):
    nombres = {columna.casefold() for columna in columnas}
    while nombre.casefold() in nombres:
        nombre += "_analizado"
    return nombre


def miles(numero):
    return f"{numero:,}".replace(",", ".")


def cantidad(numero, singular, plural):
    return f"{miles(numero)} {singular if numero == 1 else plural}"


def plegar(texto):
    """Minúsculas y sin tildes, para buscar «limpieza» aunque se escriba «LIMPIEZA» o «limpíeza»."""
    return "".join(c for c in unicodedata.normalize("NFD", texto.casefold()) if unicodedata.category(c) != "Mn")


def columnas_de_contexto(columnas, filas, indice, maximo=2):
    """Columnas breves del archivo (identificador, fecha, tema…) que ayudan a ubicar cada reseña."""
    elegidas = []
    for posicion, nombre in enumerate(columnas):
        if posicion == indice:
            continue
        valores = [fila[posicion] for fila in filas[:200]]
        if valores and max(len(valor) for valor in valores) <= 40:
            elegidas.append(posicion)
        if len(elegidas) == maximo:
            break
    return elegidas


def resaltar_negativas(valor):
    return "background-color: #FBEBD3; color: #6A3B12" if valor == "Negativa" else ""


def fila_seleccionada(evento):
    filas = evento.selection.rows if evento is not None else []
    return filas[0] if filas else None


def mostrar_etiqueta(etiqueta):
    mensaje = f"Clasificación sugerida: {etiqueta}"
    if etiqueta == "negativa":
        st.warning(mensaje, icon=":material/flag:")
    else:
        st.success(mensaje, icon=":material/check_circle:")


def reseña_individual():
    st.markdown("**Revisar una reseña**")
    st.caption("Para un comentario suelto, como uno recibido por correo o en recepción.")
    with st.form("analizar_individual", border=False):
        texto = st.text_area(
            "Texto de la reseña", height=150, max_chars=MAX_CARACTERES,
            placeholder="Escribe o pega el comentario del huésped…", label_visibility="collapsed",
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
        st.session_state.pop("temas_csv", None)
        pasos = [
            (":material/upload_file:", "Carga el archivo",
             "Un CSV con una fila de encabezados y una columna con el texto de cada reseña."),
            (":material/flag:", "Revisa lo urgente",
             "Cada reseña recibe una clasificación sugerida; filtra las negativas o busca una palabra."),
            (":material/category:", "Agrupa lo repetido",
             "Agrupa las quejas parecidas, léelas completas y descarga la selección."),
        ]
        for columna, (icono, titulo, texto) in zip(st.columns(3, gap="medium"), pasos, strict=True):
            with columna:
                st.markdown(f"{icono} **{titulo}**")
                st.caption(texto)
        st.caption("Las demás columnas del archivo, como un identificador o la fecha, se conservan en los resultados "
                   "y en la descarga. ¿Necesitas un ejemplo del formato?")
        st.download_button("Descargar plantilla", data=("\ufeff" + PLANTILLA).encode("utf-8"),
                           file_name="plantilla_resenas.csv", mime="text/csv", icon=":material/description:",
                           on_click="ignore")
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
        st.session_state.pop("temas_csv", None)
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
        mostrar_resultados(columnas, filas, indice, resultado[1], clave)


def mostrar_temas(columnas, filas, indice, etiquetas, clave):
    st.subheader("Problemas que se repiten")
    negativas = [(numero, fila) for numero, (fila, etiqueta) in
                 enumerate(zip(filas, etiquetas, strict=True), 1) if etiqueta == "negativa"]
    if not negativas:
        st.info("Ninguna reseña del archivo se clasificó como negativa.", icon=":material/check_circle:")
        return
    if len(negativas) > MAX_AGRUPACION:
        st.info("Para agrupar las quejas, carga un archivo con hasta 5.000 reseñas negativas.")
        return
    resultado = st.session_state.get("temas_csv")
    if resultado is None or resultado[0] != clave:
        st.caption(f"Reúne {cantidad(len(negativas), 'la reseña negativa', 'las reseñas negativas')} en grupos de comentarios parecidos "
                   "para ver qué asuntos se repiten y decidir qué revisar primero.")
        if st.button("Agrupar reseñas negativas", icon=":material/category:", key=f"agrupar_{clave[0]}"):
            try:
                with st.spinner("Buscando comentarios parecidos…"):
                    st.session_state["temas_csv"] = clave, agrupar_quejas(negativas, indice)
            except ErrorAnalisis as error:
                st.error(str(error))
            else:
                st.rerun()
        return

    grupos = resultado[1]
    contexto = columnas_de_contexto(columnas, filas, indice)
    opciones = {grupo["id"]: grupo for grupo in grupos}
    elegido = st.pills(
        "Grupos de quejas, de más a menos frecuente", list(opciones),
        format_func=lambda i: f"{opciones[i]['tema']} · {opciones[i]['cantidad']}",
        key=f"grupo_quejas_{clave[0]}_{indice}",
        help="Selecciona un grupo para ver solo sus quejas. Vuelve a pulsarlo para ver todas.",
    )
    st.caption("Los grupos reúnen comentarios con palabras parecidas. Lee los textos antes de decidir: "
               "una queja puede tratar varios asuntos y la cantidad no indica gravedad.")
    por_numero = {numero: grupo for grupo in grupos for numero in grupo["resenas"]}
    visibles = [(numero, fila, por_numero[numero]) for numero, fila in negativas
                if elegido is None or por_numero[numero]["id"] == elegido]
    tabla = pd.DataFrame([
        {"Reseña": numero, **{columnas[i]: fila[i] for i in contexto},
         "Grupo": grupo["tema"], "Texto": fila[indice]}
        for numero, fila, grupo in visibles
    ])
    evento = st.dataframe(
        tabla, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
        key=f"quejas_{clave[0]}_{indice}_{elegido}",
        column_config={"Texto": st.column_config.TextColumn(width="large")},
        alt="Quejas del grupo seleccionado",
    )
    seleccion = fila_seleccionada(evento)
    if seleccion is None:
        st.caption("Selecciona una fila para leer la queja completa.")
    else:
        numero, fila, grupo = visibles[seleccion]
        with st.container(border=True):
            st.markdown(f"**Queja {numero}** · {grupo['tema']}")
            st.text(fila[indice])
    cabecera = list(columnas)
    for nombre in ("sentimiento", "grupo_queja", "tema_queja"):
        cabecera.append(nombre_disponible(nombre, cabecera))
    st.download_button(
        f"Descargar {cantidad(len(visibles), 'queja', 'quejas')}", data=exportar_csv(cabecera, (
            [*fila, "negativa", str(grupo["id"]), grupo["tema"]] for _, fila, grupo in visibles
        )), file_name="quejas_agrupadas.csv", mime="text/csv", icon=":material/download:", on_click="ignore",
        key=f"descargar_quejas_{clave[0]}_{elegido}",
    )


def mostrar_resultados(columnas, filas, indice, etiquetas, clave):
    st.divider()
    st.subheader("Resultados")
    total = len(etiquetas)
    negativas = etiquetas.count("negativa")
    st.markdown(f"**{miles(negativas)} de {miles(total)} reseñas** se clasificaron como negativas "
                f"({round(100 * negativas / total)} %) y conviene revisarlas.")

    conteos = {"Todas": total, "Negativas": negativas, "Positivas": total - negativas}
    filtros = {"Todas": None, "Negativas": "negativa", "Positivas": "positiva"}
    with st.container(horizontal=True, vertical_alignment="bottom"):
        opcion = st.segmented_control(
            "Mostrar", list(filtros), default="Todas", required=True, key=f"filtro_csv_{clave[0]}",
            format_func=lambda nombre: f"{nombre} · {miles(conteos[nombre])}",
        )
        busqueda = st.text_input(
            "Buscar en el texto", placeholder="Por ejemplo: wifi, ruido, desayuno",
            key=f"buscar_{clave[0]}", width=320,
        )
    termino = plegar(busqueda.strip())
    visibles = [
        (numero, fila, etiqueta) for numero, (fila, etiqueta) in enumerate(zip(filas, etiquetas, strict=True), 1)
        if (filtros[opcion] is None or filtros[opcion] == etiqueta)
        and (not termino or termino in plegar(fila[indice]))
    ]
    if not visibles:
        st.info("Ninguna reseña coincide con el filtro y la búsqueda. Prueba con otra palabra o muestra todas.",
                icon=":material/search_off:")
    else:
        contexto = columnas_de_contexto(columnas, filas, indice)
        tabla = pd.DataFrame([
            {"Reseña": numero, **{columnas[i]: fila[i] for i in contexto},
             "Clasificación": etiqueta.capitalize(), "Texto": fila[indice]}
            for numero, fila, etiqueta in visibles
        ])
        evento = st.dataframe(
            tabla.style.map(resaltar_negativas, subset=["Clasificación"]), width="stretch", hide_index=True,
            on_select="rerun", selection_mode="single-row", key=f"tabla_{clave[0]}_{opcion}_{termino}",
            column_config={"Texto": st.column_config.TextColumn(width="large")},
            alt="Reseñas analizadas con su clasificación sugerida",
        )
        seleccion = fila_seleccionada(evento)
        if seleccion is None:
            st.caption("Selecciona una fila para leer la reseña completa.")
        else:
            numero, fila, etiqueta = visibles[seleccion]
            with st.container(border=True):
                datos = " · ".join(f"{columnas[i]}: {fila[i]}" for i in contexto)
                st.markdown(f"**Reseña {numero}**" + (f" · {datos}" if datos else ""))
                mostrar_etiqueta(etiqueta)
                st.text(fila[indice])

        nueva_columna = nombre_disponible("sentimiento", columnas)
        st.download_button(
            f"Descargar {cantidad(len(visibles), 'reseña', 'reseñas')}", data=exportar_csv(
                [*columnas, nueva_columna], ([*fila, etiqueta] for _, fila, etiqueta in visibles)
            ),
            file_name=f"resenas_{opcion.lower()}.csv", mime="text/csv", icon=":material/download:",
            key=f"descargar_csv_{clave[0]}_{opcion}_{termino}", on_click="ignore",
        )
    st.divider()
    mostrar_temas(columnas, filas, indice, etiquetas, clave)


st.set_page_config(page_title="Reseñas del hotel", page_icon=str(MARCA), layout="centered")
with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center"):
    with st.container(horizontal=True, vertical_alignment="center", gap="small", width="content"):
        st.image(str(MARCA), width=48, alt="")
        st.title("Reseñas del hotel")
    with st.popover("Revisar una reseña", icon=":material/edit_note:"):
        reseña_individual()
st.markdown("Carga las reseñas de tus huéspedes y encuentra las quejas que necesitan atención.")
cargar_archivo()
