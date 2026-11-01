"""API local de clasificación de reseñas hoteleras."""

import hashlib
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import joblib
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from sklearn.pipeline import Pipeline


RESULTADOS = Path(__file__).resolve().parent / "resultados" / "modelo"
MAX_CARACTERES = 10_000
MAX_RESENAS = 500
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


app = FastAPI(title="Análisis de reseñas hoteleras", version="1.0.0", lifespan=lifespan)


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
