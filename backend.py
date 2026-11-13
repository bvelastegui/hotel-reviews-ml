"""API local de clasificación de reseñas hoteleras.

Las operaciones de uso (`/predecir`, `/agrupar`) atienden a la interfaz del administrador.
Las operaciones de `/modelo` describen, activan o restauran pipelines que produce el cuaderno
`modelo_resenas.ipynb`; la API no entrena modelos.
"""

import hashlib
import json
import math
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline


RAIZ_RESULTADOS = Path(__file__).resolve().parent / "resultados"
RESULTADOS = RAIZ_RESULTADOS / "modelo"  # Modelo de referencia evaluado en el cuaderno.
# Cada ejecución del cuaderno lanzada desde la herramienta escribe en resultados/modelos/<id>/.
MODELOS_ENTRENADOS = RAIZ_RESULTADOS / "modelos"
MODELO_ACTIVO = RAIZ_RESULTADOS / "modelo_activo.json"
IDENTIFICADOR = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
MAX_CARACTERES = 10_000
MAX_RESENAS = 500
MAX_AGRUPACION = 5_000
# Solo se excluyen al agrupar temas; el clasificador conserva su vocabulario.
PALABRAS_FUNCIONALES = """
a al algo algun alguna algunas alguno algunos ante antes aquel aquella aqui asi aun
aunque bajo bien cada casi como cómo con contra cual cuál cuando cuándo de del desde donde dónde dos el él ella
ellas ello ellos en entre era eramos eran es esa esas ese eso esos esta estaba estaban
estamos estan están está estar estas este esto estos estuvo fue fueron fui fuimos ha habia había hace
hacen hacia han hasta hay hotel hoteles la las le les lo los mas me mi mis mismo
mucha muchas mucho muchos muy nada ni no nos nosotros nuestra nuestro nunca o otra
otras otro otros para pero poco por porque pues que quien se sea ser si sin sobre
solo sólo somos son su sus tal tanto te tiene tienen todo todos tras tu tus un una unas
uno unos vez ya yo
algún aquí así aún más mí sí tú
""".split()
Sentimiento = Literal["negativa", "positiva"]
TextoResena = Annotated[
    str, StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=MAX_CARACTERES)
]
ETIQUETAS: dict[int, Sentimiento] = {0: "negativa", 1: "positiva"}


class Resena(BaseModel):
    model_config = ConfigDict(extra="forbid")
    texto: TextoResena


class Lote(BaseModel):
    model_config = ConfigDict(extra="forbid")
    resenas: Annotated[list[Resena], Field(min_length=1, max_length=MAX_RESENAS)]


class Clasificacion(BaseModel):
    sentimiento: Sentimiento


class Clasificaciones(BaseModel):
    resultados: list[Clasificacion]


class ResenaNumerada(Resena):
    numero: Annotated[int, Field(strict=True, ge=1)]


class LoteNegativas(BaseModel):
    model_config = ConfigDict(extra="forbid")
    resenas: Annotated[list[ResenaNumerada], Field(max_length=MAX_AGRUPACION)]

    @model_validator(mode="after")
    def validar_numeros(self):
        if len({resena.numero for resena in self.resenas}) != len(self.resenas):
            raise ValueError("Cada reseña debe tener un número distinto.")
        if sum(len(resena.texto) for resena in self.resenas) > 10_000_000:
            raise ValueError("El texto total supera los 10 millones de caracteres.")
        return self


class GrupoNegativas(BaseModel):
    id: int
    tema: str
    palabras: list[str]
    cantidad: int
    resenas: list[int]


class AgrupacionNegativas(BaseModel):
    total: int
    grupos: list[GrupoNegativas]


def agrupar_negativas(lote: LoteNegativas) -> AgrupacionNegativas:
    """Agrupa exclusivamente el lote recibido; no modifica el modelo de sentimiento."""
    if not lote.resenas:
        return AgrupacionNegativas(total=0, grupos=[])
    resenas = sorted(lote.resenas, key=lambda resena: resena.numero)
    vectorizador = TfidfVectorizer(
        stop_words=PALABRAS_FUNCIONALES,
        token_pattern=r"(?u)\b[^\W\d_]{3,}\b", max_features=5_000, sublinear_tf=True,
    )
    try:
        matriz = vectorizador.fit_transform(resena.texto for resena in resenas)
    except ValueError as error:
        if "empty vocabulary" not in str(error):
            raise
        return AgrupacionNegativas(total=len(resenas), grupos=[GrupoNegativas(
            id=1, tema="Sin términos suficientes", palabras=[], cantidad=len(resenas),
            resenas=[resena.numero for resena in resenas],
        )])

    matriz.sort_indices()
    utilizables = np.flatnonzero(matriz.getnnz(axis=1))
    sin_terminos = np.flatnonzero(matriz.getnnz(axis=1) == 0)
    vectores = matriz[utilizables]
    distintos = len({
        (tuple(vectores[i].indices), tuple(vectores[i].data.round(12)))
        for i in range(vectores.shape[0])
    })
    # Regla exploratoria acotada: hasta cinco grupos, sin forzar grupos vacíos.
    cantidad = min(5, distintos, math.ceil(math.sqrt(len(utilizables) / 2)))
    etiquetas = KMeans(n_clusters=cantidad, n_init=10, random_state=42).fit_predict(vectores)
    terminos = vectorizador.get_feature_names_out()
    grupos = []
    for etiqueta in sorted(set(etiquetas)):
        indices = utilizables[etiquetas == etiqueta]
        pesos = np.asarray(matriz[indices].mean(axis=0)).ravel()
        orden = sorted(np.flatnonzero(pesos), key=lambda i: (-pesos[i], terminos[i]))[:3]
        palabras = [str(terminos[i]) for i in orden]
        grupos.append(GrupoNegativas(
            id=0, tema=" · ".join(palabras), palabras=palabras, cantidad=len(indices),
            resenas=[resenas[i].numero for i in indices],
        ))
    if sin_terminos.size:
        grupos.append(GrupoNegativas(
            id=0, tema="Sin términos suficientes", palabras=[], cantidad=len(sin_terminos),
            resenas=[resenas[i].numero for i in sin_terminos],
        ))
    grupos.sort(key=lambda grupo: (-grupo.cantidad, grupo.resenas[0]))
    for numero, grupo in enumerate(grupos, 1):
        grupo.id = numero
    return AgrupacionNegativas(total=len(resenas), grupos=grupos)


def carpeta_ejecucion(identificador: str) -> Path:
    """Carpeta del pipeline exportado por una ejecución del cuaderno."""
    if not IDENTIFICADOR.match(identificador):
        raise ValueError("Identificador de ejecución no válido.")
    return MODELOS_ENTRENADOS / identificador / "modelo"


def carpeta_activa() -> Path:
    """Carpeta del modelo que sirve la API; sin selección, el modelo de referencia."""
    try:
        seleccion = json.loads(MODELO_ACTIVO.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return RESULTADOS
    try:
        carpeta = carpeta_ejecucion(str(seleccion.get("id", "")))
    except ValueError as error:
        raise RuntimeError("modelo_activo.json no es válido. Restaura el modelo de referencia.") from error
    if not carpeta.is_dir():
        raise RuntimeError("modelo_activo.json apunta a un modelo inexistente. Restaura el modelo de referencia.")
    return carpeta


def cargar_pipeline(carpeta: Path | None = None) -> tuple[Pipeline, dict]:
    carpeta = carpeta or carpeta_activa()
    ruta = carpeta / "sentimiento.joblib"
    try:
        evidencia = json.loads((carpeta / "metricas.json").read_text(encoding="utf-8"))
        with ruta.open("rb") as archivo:
            if hashlib.file_digest(archivo, "sha256").hexdigest() != evidencia["pipeline"]["sha256"]:
                raise RuntimeError("La huella del pipeline difiere del registro de entrenamiento.")
            archivo.seek(0)
            pipeline = joblib.load(archivo)
    except FileNotFoundError as error:
        raise RuntimeError("Faltan las evidencias del modelo. Ejecuta modelo_resenas.ipynb.") from error
    if not isinstance(pipeline, Pipeline) or list(getattr(pipeline, "classes_", [])) != [0, 1]:
        raise RuntimeError("El pipeline debe estar entrenado con las clases binarias 0 y 1.")
    return pipeline, evidencia


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    carpeta = carpeta_activa()
    app.state.pipeline, app.state.evidencia = cargar_pipeline(carpeta)
    app.state.id_activo = None if carpeta == RESULTADOS else carpeta.parent.name
    yield


app = FastAPI(title="Análisis de reseñas hoteleras", version="1.2.0", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def error_entrada(_request: Request, error: RequestValidationError) -> JSONResponse:
    detalles = [
        {campo: item[campo] for campo in ("loc", "msg", "type")}
        for item in error.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": detalles})


@app.get("/salud")
def salud() -> dict[str, str]:
    return {"estado": "listo"}


@app.post("/predecir")
def predecir(resena: Resena, request: Request) -> Clasificacion:
    etiqueta = request.app.state.pipeline.predict([resena.texto])[0]
    return Clasificacion(sentimiento=ETIQUETAS[int(etiqueta)])


@app.post("/predecir/lote")
def predecir_lote(lote: Lote, request: Request) -> Clasificaciones:
    etiquetas = request.app.state.pipeline.predict([resena.texto for resena in lote.resenas])
    return Clasificaciones(
        resultados=[Clasificacion(sentimiento=ETIQUETAS[int(etiqueta)]) for etiqueta in etiquetas]
    )


@app.post("/agrupar/negativas")
def temas_negativas(lote: LoteNegativas) -> AgrupacionNegativas:
    """Recibe reseñas ya clasificadas como negativas, con su número de fila original."""
    return agrupar_negativas(lote)


# ---------------------------------------------------------------------------
# Modelos producidos por el cuaderno: descripción, activación y restauración
# ---------------------------------------------------------------------------

class MetricasClase(BaseModel):
    precision: float
    recall: float
    f1: float
    soporte: int


class Metricas(BaseModel):
    f1_macro: float
    exactitud: float
    negativa: MetricasClase
    positiva: MetricasClase
    matriz_confusion: list[list[int]]


class DescripcionModelo(BaseModel):
    algoritmo: str
    nombre: str
    parametros: dict
    tfidf: dict
    umbral: float | None


class ModeloServido(BaseModel):
    origen: Literal["referencia", "herramienta"]
    id: str | None
    modelo: DescripcionModelo | None
    validacion: Metricas | None
    prueba: Metricas | None


class Activacion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: Annotated[str, StringConstraints(strict=True, pattern=r"^[A-Za-z0-9_-]{1,64}$")]


def convertir_metricas(datos: dict | None) -> Metricas | None:
    if not datos:
        return None
    clase = lambda d: MetricasClase(precision=d["precision"], recall=d["recall"],  # noqa: E731
                                    f1=d["f1-score"], soporte=int(d["support"]))
    return Metricas(f1_macro=datos["f1_macro"], exactitud=datos["exactitud"],
                    negativa=clase(datos["negativa"]), positiva=clase(datos["positiva"]),
                    matriz_confusion=datos["matriz_confusion"])


def describir_servido(app_state) -> ModeloServido:
    evidencia = app_state.evidencia
    descripcion = evidencia.get("modelo")
    return ModeloServido(
        origen="herramienta" if app_state.id_activo else "referencia", id=app_state.id_activo,
        modelo=DescripcionModelo(**descripcion) if descripcion else None,
        validacion=convertir_metricas(evidencia.get("evaluacion_validacion")),
        prueba=convertir_metricas(evidencia.get("evaluacion_prueba")),
    )


@app.get("/modelo")
def modelo_servido(request: Request) -> ModeloServido:
    """Describe el modelo que responde a la interfaz y las métricas que registró el cuaderno."""
    return describir_servido(request.app.state)


@app.post("/modelo/activar")
def activar(activacion: Activacion, request: Request) -> ModeloServido:
    """Pone en uso el pipeline de una ejecución del cuaderno, tras verificar su huella."""
    carpeta = carpeta_ejecucion(activacion.id)
    if not carpeta.is_dir():
        raise HTTPException(status_code=404, detail="No existe esa ejecución del cuaderno.")
    try:
        pipeline, evidencia = cargar_pipeline(carpeta)
    except RuntimeError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    temporal = MODELO_ACTIVO.with_suffix(".tmp")
    temporal.write_text(json.dumps({"id": activacion.id}) + "\n", encoding="utf-8")
    temporal.replace(MODELO_ACTIVO)
    estado = request.app.state
    estado.pipeline, estado.evidencia, estado.id_activo = pipeline, evidencia, activacion.id
    return describir_servido(estado)


@app.post("/modelo/restaurar")
def restaurar(request: Request) -> ModeloServido:
    """Vuelve al modelo de referencia exportado y evaluado por el cuaderno."""
    pipeline, evidencia = cargar_pipeline(RESULTADOS)
    MODELO_ACTIVO.unlink(missing_ok=True)
    estado = request.app.state
    estado.pipeline, estado.evidencia, estado.id_activo = pipeline, evidencia, None
    return describir_servido(estado)
