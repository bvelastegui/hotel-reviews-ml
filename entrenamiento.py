"""Herramienta técnica para configurar, entrenar y activar el clasificador de reseñas.

No contiene código de entrenamiento: recoge los parámetros, ejecuta el cuaderno
`modelo_resenas.ipynb` con papermill y muestra las métricas que el cuaderno registra.
Cada ejecución queda en `resultados/modelos/<id>/` junto con el cuaderno ejecutado.
Está separada de la interfaz del administrador (`frontend.py`).
"""

import json
import os
import time
import uuid
from pathlib import Path

import pandas as pd
import papermill
import requests
import streamlit as st


BASE = Path(__file__).resolve().parent
CUADERNO = BASE / "modelo_resenas.ipynb"
EJECUCIONES = BASE / "resultados" / "modelos"
API_URL = os.environ.get("RESENAS_API_URL", "http://127.0.0.1:8000").rstrip("/")
ALGORITMOS = {
    "svm": "SVM lineal",
    "naive_bayes": "Complement Naive Bayes",
    "logistica": "Regresión logística",
    "knn": "KNN (distancia coseno)",
    "arbol": "Árbol de decisión",
    "mayoritario": "Referencia mayoritaria",
}
NOMBRES_PARAMETROS = {
    "C": "C", "class_weight": "Pesos de clase", "alpha": "Alpha", "n_neighbors": "Vecinos (k)",
    "weights": "Ponderación", "max_depth": "Profundidad máxima",
}


class ErrorApi(Exception):
    """Problema de comunicación con la API que la herramienta puede mostrar."""


def llamar(metodo, ruta, datos=None, espera=30):
    try:
        respuesta = requests.request(metodo, f"{API_URL}{ruta}", json=datos, timeout=(3, espera))
    except requests.ConnectionError as error:
        raise ErrorApi(f"No hay conexión con la API en {API_URL}. Inicia backend.py.") from error
    except requests.Timeout as error:
        raise ErrorApi("La API tardó demasiado en responder.") from error
    if respuesta.status_code >= 400:
        try:
            detalle = respuesta.json().get("detail")
        except ValueError:
            detalle = None
        if isinstance(detalle, list):
            detalle = "; ".join(f'{".".join(map(str, d.get("loc", [])[1:]))}: {d.get("msg")}' for d in detalle)
        raise ErrorApi(detalle or f"La API respondió con el código {respuesta.status_code}.")
    return respuesta.json()


def porcentaje(valor):
    return f"{valor:.1%}"


def valor_parametro(clave, valor):
    if clave == "class_weight":
        return "balanceados" if valor == "balanced" else "sin ponderar"
    if clave == "weights":
        return "por distancia" if valor == "distance" else "uniforme"
    return "sin límite" if valor is None else valor


def miles(numero):
    return f"{numero:,}".replace(",", ".")


def describir(modelo):
    partes = [f"{NOMBRES_PARAMETROS.get(k, k)}: {valor_parametro(k, v)}" for k, v in modelo["parametros"].items()]
    tfidf = modelo["tfidf"]
    ngramas = "1" if tfidf["ngram_range"][1] == 1 else f"1–{tfidf['ngram_range'][1]}"
    partes.append(f"n-gramas {ngramas}, min_df {tfidf['min_df']}, máx. {miles(tfidf['max_features'])} términos")
    partes.append("regla predeterminada" if modelo["umbral"] is None else f"umbral = {modelo['umbral']:.3f}")
    return " · ".join(partes)


def tabla_matriz(matriz):
    return pd.DataFrame(matriz, index=["Real negativa", "Real positiva"],
                        columns=["Predicha negativa", "Predicha positiva"])


def tabla_clases(metricas):
    return pd.DataFrame([
        {"Clase": nombre, "Precisión": porcentaje(metricas[clave]["precision"]),
         "Recall": porcentaje(metricas[clave]["recall"]), "F1": f'{metricas[clave]["f1"]:.4f}',
         "Reseñas": metricas[clave]["soporte"]}
        for clave, nombre in (("negativa", "Negativa"), ("positiva", "Positiva"))
    ])


class ErrorCuaderno(Exception):
    """La ejecución del cuaderno no terminó correctamente."""


def convertir(datos):
    """Métricas del cuaderno (formato de classification_report) al formato de la herramienta."""
    if not datos:
        return None
    clase = lambda d: {"precision": d["precision"], "recall": d["recall"],  # noqa: E731
                       "f1": d["f1-score"], "soporte": int(d["support"])}
    return {"f1_macro": datos["f1_macro"], "exactitud": datos["exactitud"],
            "negativa": clase(datos["negativa"]), "positiva": clase(datos["positiva"]),
            "matriz_confusion": datos["matriz_confusion"]}


def parametros_cuaderno(datos, carpeta):
    """Traduce el formulario a los parámetros de la celda `parameters` del cuaderno."""
    algoritmo = datos["algoritmo"]
    pesos = "balanced" if datos.get("pesos_balanceados") else None
    parametros = {
        "FAMILIAS": [] if algoritmo == "mayoritario" else [algoritmo],
        "NGRAMAS_MAX": datos["ngramas_max"], "MIN_DF": datos["min_df"], "MAX_TERMINOS": datos["max_caracteristicas"],
        "META_RECALL_NEGATIVAS": datos["meta_recall_negativas"],
        "VALIDACION_CRUZADA": datos["validacion_cruzada"],
        "EVALUAR_PRUEBA": False,  # la prueba reservada solo mide el modelo de referencia
        "EXPORTAR_MODELO": True, "EXPLORAR_GRUPOS": False,
        "CARPETA_SALIDA": str(carpeta.relative_to(BASE)),
    }
    if algoritmo == "svm":
        parametros |= {"C_SVM": [datos["c"]], "PESOS_SVM": [pesos]}
    elif algoritmo == "logistica":
        parametros |= {"C_LOGISTICA": [datos["c"]], "PESOS_LOGISTICA": [pesos]}
    elif algoritmo == "naive_bayes":
        parametros |= {"ALPHA_NAIVE_BAYES": [datos["alpha"]]}
    elif algoritmo == "knn":
        parametros |= {"K_VECINOS": [datos["vecinos"]]}
    elif algoritmo == "arbol":
        parametros |= {"PROFUNDIDAD_ARBOL": [datos["profundidad_maxima"]], "PESOS_ARBOL": [pesos]}
    return parametros


def ejecutar_cuaderno(datos):
    identificador = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    carpeta = EJECUCIONES / identificador
    carpeta.mkdir(parents=True)
    inicio = time.perf_counter()
    try:
        papermill.execute_notebook(
            str(CUADERNO), str(carpeta / "modelo_resenas.ipynb"), parameters=parametros_cuaderno(datos, carpeta),
            cwd=str(BASE), kernel_name="python3", progress_bar=False, log_output=False,
        )
    except papermill.PapermillExecutionError as error:
        raise ErrorCuaderno(f"El cuaderno se detuvo con {error.ename}: {error.evalue}") from error
    metricas = json.loads((carpeta / "modelo" / "metricas.json").read_text(encoding="utf-8"))
    particiones = json.loads((BASE / "datos" / "particiones.json").read_text(encoding="utf-8"))["particiones"]
    cruzada = None
    if datos["validacion_cruzada"]:
        revision = json.loads((carpeta / "revision_materia.json").read_text(encoding="utf-8"))
        resumen = revision["resumen"]["modelo_de_la_aplicacion"]["f1_macro"]
        cruzada = {"pliegues": revision["resultados_por_pliegue"]["modelo_de_la_aplicacion"],
                   "f1_macro_media": resumen["media"], "f1_macro_desviacion": resumen["desviacion_estandar_muestral"]}
    umbral = metricas["umbral"]["umbral"]
    aviso = None
    if datos["meta_recall_negativas"] is not None and umbral is None and datos["algoritmo"] != "mayoritario":
        aviso = ("Ningún umbral alcanza la meta de recall sin perder más de 0,02 de F1 macro; "
                 "se conserva la regla predeterminada.")
    return {
        "id": identificador, "modelo": metricas["modelo"],
        "n_entrenamiento": particiones["entrenamiento"]["total"], "n_validacion": particiones["validacion"]["total"],
        "validacion": convertir(metricas["evaluacion_validacion"]),
        "validacion_sin_umbral": None if umbral is None else convertir(metricas["evaluacion_validacion_sin_umbral"]),
        "validacion_cruzada": cruzada, "aviso": aviso, "segundos": round(time.perf_counter() - inicio, 1),
        "cuaderno": carpeta / "modelo_resenas.ipynb",
    }


def mostrar_modelo_activo(activo):
    modelo = activo["modelo"]
    origen = ("modelo de referencia del cuaderno" if activo["origen"] == "referencia"
              else f"ejecución {activo['id']} del cuaderno")
    st.markdown(f"**{modelo['nombre']}**, {origen}.")
    st.caption(describir(modelo))
    validacion, prueba = activo.get("validacion"), activo.get("prueba")
    with st.container(horizontal=True):
        if validacion:
            st.metric("F1 macro en validación", f'{validacion["f1_macro"]:.4f}', border=True)
            st.metric("Recall de negativas en validación", porcentaje(validacion["negativa"]["recall"]), border=True)
        if prueba:
            st.metric("F1 macro en prueba reservada", f'{prueba["f1_macro"]:.4f}', border=True)
            st.metric("Recall de negativas en prueba", porcentaje(prueba["negativa"]["recall"]), border=True)
    if prueba is None:
        st.caption("La prueba reservada solo evalúa el modelo de referencia del cuaderno.")


def formulario(algoritmo):
    with st.form("configuracion"):
        datos = {"algoritmo": algoritmo}
        st.markdown("**Parámetros del algoritmo**")
        if algoritmo in ("svm", "logistica"):
            datos["c"] = st.number_input(
                "C (inverso de la regularización)", min_value=0.01, max_value=100.0, value=2.0, step=0.5, key="c",
                help="Valores altos ajustan más el modelo a los datos; valores bajos lo regularizan más.")
            datos["pesos_balanceados"] = st.toggle("Pesos de clase balanceados", value=True, key="pesos",
                                                   help="Compensa que solo el 18 % de las reseñas son negativas.")
        elif algoritmo == "naive_bayes":
            datos["alpha"] = st.number_input("Alpha (suavizado)", min_value=0.01, max_value=10.0, value=0.5, step=0.1,
                                            key="alpha")
        elif algoritmo == "knn":
            datos["vecinos"] = st.slider("Vecinos (k)", min_value=1, max_value=99, value=15, step=2, key="vecinos")
        elif algoritmo == "arbol":
            sin_limite = st.toggle("Profundidad sin límite", value=False)
            profundidad = st.number_input("Profundidad máxima", min_value=1, max_value=500, value=30,
                                          help="Se ignora si la profundidad no tiene límite.")
            datos["profundidad_maxima"] = None if sin_limite else int(profundidad)
            datos["pesos_balanceados"] = st.toggle("Pesos de clase balanceados", value=True)
        else:
            st.caption("La referencia mayoritaria no tiene parámetros: siempre predice la clase más frecuente.")

        st.markdown("**Representación TF-IDF**")
        columnas = st.columns(3)
        datos["ngramas_max"] = columnas[0].segmented_control(
            "N-gramas", [1, 2, 3], default=2, format_func=lambda n: f"1–{n}" if n > 1 else "1",
            required=True)
        datos["min_df"] = columnas[1].number_input("Frecuencia mínima (min_df)", min_value=1, max_value=50, value=2)
        datos["max_caracteristicas"] = columnas[2].number_input(
            "Máximo de términos", min_value=1_000, max_value=200_000, value=40_000, step=5_000)

        st.markdown("**Umbral y validación**")
        ajustar = st.toggle("Ajustar el umbral para no perder quejas", value=algoritmo != "mayoritario",
                            disabled=algoritmo == "mayoritario")
        meta = st.slider("Meta de recall de negativas en validación", min_value=0.50, max_value=0.99,
                         value=0.90, step=0.01, format="%.2f", disabled=algoritmo == "mayoritario")
        datos["meta_recall_negativas"] = meta if ajustar and algoritmo != "mayoritario" else None
        datos["validacion_cruzada"] = st.toggle("Calcular validación cruzada de 5 pliegues por grupos", value=False,
                                                key="cruzada",
                                                help="Repite el ajuste cinco veces dentro de entrenamiento.")
        enviado = st.form_submit_button("Entrenar", type="primary", icon=":material/model_training:", key="entrenar")
    return datos if enviado else None


def mostrar_resultado(resultado, activo):
    validacion = resultado["validacion"]
    referencia = (activo or {}).get("validacion")

    def delta(valor, clave_referencia, puntos=True):
        if not referencia:
            return None
        diferencia = valor - clave_referencia(referencia)
        texto = f"{diferencia * 100:+.1f} pp" if puntos else f"{diferencia:+.4f}"
        return 0 if texto.lstrip("+-").rstrip(" p") in ("0.0", "0.0000") else texto

    st.markdown(f"**{resultado['modelo']['nombre']}** · {describir(resultado['modelo'])}")
    st.caption(f"El cuaderno entrenó con {miles(resultado['n_entrenamiento'])} reseñas y evaluó con "
               f"{miles(resultado['n_validacion'])} de validación en {resultado['segundos']} s.")
    if resultado.get("aviso"):
        st.info(resultado["aviso"], icon=":material/info:")
    columnas = st.columns(4)
    with columnas[0]:
        st.metric("F1 macro", f'{validacion["f1_macro"]:.4f}',
                  delta(validacion["f1_macro"], lambda r: r["f1_macro"], puntos=False),
                  delta_description="vs. en uso" if referencia else None, border=True)
    with columnas[1]:
        st.metric("Exactitud", porcentaje(validacion["exactitud"]),
                  delta(validacion["exactitud"], lambda r: r["exactitud"]),
                  delta_description="vs. en uso" if referencia else None, border=True)
    with columnas[2]:
        st.metric("Recall de negativas", porcentaje(validacion["negativa"]["recall"]),
                  delta(validacion["negativa"]["recall"], lambda r: r["negativa"]["recall"]),
                  delta_description="vs. en uso" if referencia else None, border=True)
    with columnas[3]:
        st.metric("Precisión de negativas", porcentaje(validacion["negativa"]["precision"]),
                  delta(validacion["negativa"]["precision"], lambda r: r["negativa"]["precision"]),
                  delta_description="vs. en uso" if referencia else None, border=True)

    izquierda, derecha = st.columns(2)
    with izquierda, st.container(border=True):
        st.markdown("**Matriz de confusión en validación**")
        st.table(tabla_matriz(validacion["matriz_confusion"]))
        st.caption("Filas: clase real. Columnas: clase predicha. "
                   f"{validacion['matriz_confusion'][0][1]} quejas pasarían como positivas.")
    with derecha, st.container(border=True):
        st.markdown("**Métricas por clase**")
        st.table(tabla_clases(validacion))
        if resultado.get("validacion_sin_umbral"):
            previo = resultado["validacion_sin_umbral"]
            st.caption(f"Sin ajuste de umbral: F1 macro {previo['f1_macro']:.4f}, recall de negativas "
                       f"{porcentaje(previo['negativa']['recall'])}.")

    if cruzada := resultado.get("validacion_cruzada"):
        with st.container(border=True):
            st.markdown(f"**Validación cruzada:** F1 macro {cruzada['f1_macro_media']:.4f} "
                        f"± {cruzada['f1_macro_desviacion']:.4f}")
            pliegues = pd.DataFrame(cruzada["pliegues"]).set_index("pliegue")
            pliegues.index = [f"Pliegue {i}" for i in pliegues.index]
            st.bar_chart(pliegues[["f1_macro", "recall_negativas"]].rename(
                columns={"f1_macro": "F1 macro", "recall_negativas": "Recall de negativas"}),
                stack=False, y_label="Valor", alt="F1 macro y recall de negativas por pliegue")


st.set_page_config(page_title="Entrenamiento del modelo", page_icon=":material/model_training:", layout="wide")
st.title("Entrenamiento del modelo")
st.caption("Herramienta técnica del equipo: configura parámetros, ejecuta el cuaderno de entrenamiento, "
           "compara métricas en validación y decide qué modelo usa la aplicación del hotel.")

try:
    activo = llamar("GET", "/modelo")
except ErrorApi as error:
    st.error(str(error), title="API no disponible")
    st.stop()

with st.container(border=True):
    st.subheader("Modelo en uso")
    mostrar_modelo_activo(activo)
    if activo["origen"] == "herramienta":
        if st.button("Restaurar el modelo de referencia", icon=":material/restore:", key="restaurar"):
            try:
                llamar("POST", "/modelo/restaurar")
            except ErrorApi as error:
                st.error(str(error))
            else:
                st.toast("Se restauró el modelo de referencia.", icon=":material/check_circle:")
                st.rerun()

configuracion, resultados = st.columns([2, 3], gap="large")
with configuracion:
    st.subheader("Configurar entrenamiento")
    algoritmo = st.selectbox("Algoritmo", list(ALGORITMOS), format_func=ALGORITMOS.get, key="algoritmo")
    datos = formulario(algoritmo)
    if datos is not None:
        st.session_state.pop("entrenamiento", None)
        try:
            with st.spinner("Ejecutando el cuaderno modelo_resenas.ipynb con estos parámetros…"):
                st.session_state["entrenamiento"] = ejecutar_cuaderno(datos)
        except ErrorCuaderno as error:
            st.error(str(error), title="El entrenamiento no terminó")

with resultados:
    st.subheader("Resultado del entrenamiento")
    resultado = st.session_state.get("entrenamiento")
    if resultado is None:
        st.info("Configura los parámetros y pulsa **Entrenar** para ver las métricas en validación.",
                icon=":material/tune:")
    else:
        mostrar_resultado(resultado, activo)
        cuaderno = resultado["cuaderno"]
        st.download_button("Descargar el cuaderno ejecutado", data=cuaderno.read_bytes(),
                           file_name=f"modelo_resenas_{resultado['id']}.ipynb", mime="application/x-ipynb+json",
                           icon=":material/description:", on_click="ignore", key=f"cuaderno_{resultado['id']}")
        st.caption(f"Evidencia guardada en {cuaderno.parent.relative_to(BASE)}/")
        if activo.get("id") == resultado["id"]:
            st.success("Este modelo ya responde a la aplicación del hotel.", icon=":material/check_circle:")
        elif st.button("Usar este modelo en la aplicación", type="primary", icon=":material/rocket_launch:",
                           key="activar"):
            try:
                llamar("POST", "/modelo/activar", {"id": resultado["id"]})
            except ErrorApi as error:
                st.error(str(error))
            else:
                st.toast("El modelo entrenado ya responde a la aplicación.", icon=":material/check_circle:")
                st.rerun()
