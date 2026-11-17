"""Convierte un dataset de imágenes/videos en landmarks (datasets/procesado/).

Estructura esperada de la carpeta de entrada (una subcarpeta por seña):
    mi_dataset/
        hola/   foto1.jpg  video1.mp4 ...
        casa/   ...

Genera datasets/procesado/<seña>/<archivo>.npy (un vector de 84 valores por frame
con mano detectada) y agrega filas a datasets/etiquetas.csv.

Uso:
    python scripts/extraer_landmarks.py ruta/a/mi_dataset
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentes.agente_detector_manos import DetectorManos  # noqa: E402
from agentes.agente_normalizador import vector_desde_manos  # noqa: E402
from utils.constantes import RAIZ  # noqa: E402
from utils.io_helpers import agregar_fila_csv, cargar_config, guardar_vector  # noqa: E402

EXT_IMAGEN = {".jpg", ".jpeg", ".png", ".bmp"}
EXT_VIDEO = {".mp4", ".avi", ".mov", ".mkv"}


def frames_de(archivo: Path):
    if archivo.suffix.lower() in EXT_IMAGEN:
        img = cv2.imread(str(archivo))
        if img is not None:
            yield img
    elif archivo.suffix.lower() in EXT_VIDEO:
        cap = cv2.VideoCapture(str(archivo))
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
        cap.release()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("entrada", type=Path)
    args = parser.parse_args()

    cfg = cargar_config()
    detector = DetectorManos(cfg["deteccion"])
    salida = RAIZ / "datasets" / "procesado"
    etiquetas = RAIZ / "datasets" / "etiquetas.csv"
    total = 0

    for carpeta in sorted(p for p in args.entrada.iterdir() if p.is_dir()):
        for archivo in sorted(carpeta.iterdir()):
            vectores = []
            for frame in frames_de(archivo):
                manos = detector.detectar(frame)
                if manos:
                    vectores.append(vector_desde_manos(
                        [m["puntos"] for m in manos], cfg["normalizacion"]["rotar"]))
            if not vectores:
                print(f"  sin manos: {archivo.name}")
                continue
            ruta = salida / carpeta.name / f"{archivo.stem}.npy"
            guardar_vector(ruta, np.vstack(vectores))
            agregar_fila_csv(etiquetas, ["label", "path_landmarks"],
                             [carpeta.name, str(ruta.relative_to(RAIZ))])
            total += len(vectores)
            print(f"  {carpeta.name}/{archivo.name}: {len(vectores)} vectores")

    detector.cerrar()
    print(f"Listo. {total} vectores guardados en {salida}")


if __name__ == "__main__":
    main()
