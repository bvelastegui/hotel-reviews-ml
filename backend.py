"""API local de clasificación de reseñas hoteleras."""

import hashlib
import json
import math
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import joblib
import numpy as np
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline


RESULTADOS = Path(__file__).resolve().parent / "resultados" / "modelo"
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


def cargar_pipeline() -> Pipeline:
    ruta = RESULTADOS / "sentimiento.joblib"
    try:
        evidencia = json.loads((RESULTADOS / "metricas.json").read_text(encoding="utf-8"))
        with ruta.open("rb") as archivo:
            if hashlib.file_digest(archivo, "sha256").hexdigest() != evidencia["pipeline"]["sha256"]:
                raise RuntimeError("La huella del pipeline difiere del registro de entrenamiento.")
            archivo.seek(0)
            pipeline = joblib.load(archivo)
    except FileNotFoundError as error:
        raise RuntimeError("Faltan las evidencias del modelo. Ejecuta modelo_resenas.ipynb.") from error
    if not isinstance(pipeline, Pipeline) or list(getattr(pipeline, "classes_", [])) != [0, 1]:
        raise RuntimeError("El pipeline debe estar entrenado con las clases binarias 0 y 1.")
    return pipeline


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.pipeline = cargar_pipeline()
    yield


app = FastAPI(title="Análisis de reseñas hoteleras", version="1.1.0", lifespan=lifespan)


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
