"""Inicia la API, la aplicación del hotel y la herramienta de entrenamiento.

Uso, desde la carpeta del proyecto y con el entorno virtual activo:

    python iniciar.py                 # API + aplicación + herramienta de entrenamiento
    python iniciar.py --sin-entrenamiento

Comprueba el modelo antes de arrancar, espera a que la API responda en /salud y
detiene todos los servicios con Ctrl+C.
"""

import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parent
HOST = "127.0.0.1"


def esperar_api(url, proceso, limite=60):
    inicio = time.monotonic()
    while time.monotonic() - inicio < limite:
        if proceso.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"{url}/salud", timeout=2) as respuesta:
                if respuesta.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(0.5)
    return False


def detener(procesos):
    for proceso in procesos:
        if proceso.poll() is None:
            proceso.send_signal(signal.SIGINT if os.name != "nt" else signal.CTRL_BREAK_EVENT)
    fin = time.monotonic() + 10
    for proceso in procesos:
        try:
            proceso.wait(timeout=max(0.1, fin - time.monotonic()))
        except subprocess.TimeoutExpired:
            proceso.kill()


def main():
    argumentos = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    argumentos.add_argument("--puerto-api", type=int, default=8000)
    argumentos.add_argument("--puerto-app", type=int, default=8501)
    argumentos.add_argument("--puerto-entrenamiento", type=int, default=8502)
    argumentos.add_argument("--sin-entrenamiento", action="store_true",
                            help="No inicia la herramienta de entrenamiento")
    opciones = argumentos.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    def interrumpir(_senal, _marco):
        raise KeyboardInterrupt

    # Ctrl+C y una terminación externa (SIGTERM) detienen todos los servicios por igual.
    signal.signal(signal.SIGINT, interrumpir)
    signal.signal(signal.SIGTERM, interrumpir)

    modelo = BASE / "resultados" / "modelo"
    if not (modelo / "sentimiento.joblib").exists() or not (modelo / "metricas.json").exists():
        sys.exit("Falta el modelo. Ejecuta primero modelo_resenas.ipynb para generarlo.")

    api_url = f"http://{HOST}:{opciones.puerto_api}"
    entorno = {**os.environ, "RESENAS_API_URL": api_url}
    extra = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {}
    procesos = []
    try:
        print(f"Iniciando la API en {api_url} …")
        api = subprocess.Popen([sys.executable, "-m", "uvicorn", "backend:app", "--host", HOST,
                                "--port", str(opciones.puerto_api)], cwd=BASE, env=entorno, **extra)
        procesos.append(api)
        if not esperar_api(api_url, api):
            sys.exit("La API no respondió. Revisa los mensajes anteriores.")
        print("API lista.")

        streamlit = [sys.executable, "-m", "streamlit", "run", "--server.headless=true",
                     f"--server.address={HOST}"]
        procesos.append(subprocess.Popen([*streamlit, f"--server.port={opciones.puerto_app}", "frontend.py"],
                                         cwd=BASE, env=entorno, **extra))
        enlaces = [("Aplicación del hotel", f"http://{HOST}:{opciones.puerto_app}")]
        if not opciones.sin_entrenamiento:
            procesos.append(subprocess.Popen(
                [*streamlit, f"--server.port={opciones.puerto_entrenamiento}", "entrenamiento.py"],
                cwd=BASE, env=entorno, **extra))
            enlaces.append(("Herramienta de entrenamiento", f"http://{HOST}:{opciones.puerto_entrenamiento}"))
        enlaces.append(("Documentación de la API", f"{api_url}/docs"))

        print("\nServicios en ejecución (Ctrl+C para detener):")
        for nombre, url in enlaces:
            print(f"  {nombre:<30} {url}")
        while all(proceso.poll() is None for proceso in procesos):
            time.sleep(1)
        print("Un servicio se detuvo; se cierran los demás.")
    except KeyboardInterrupt:
        print("\nDeteniendo servicios …")
    finally:
        detener(procesos)


if __name__ == "__main__":
    main()
