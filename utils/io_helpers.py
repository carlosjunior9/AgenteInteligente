"""Utilidades de lectura/escritura: config, .npy, .json, .csv y marcas de tiempo."""
import csv
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from utils.constantes import RAIZ


def cargar_config(ruta=None) -> dict:
    """Lee config.yaml y devuelve un diccionario con rutas absolutas resueltas."""
    ruta = Path(ruta) if ruta else RAIZ / "config.yaml"
    with open(ruta, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_raiz"] = str(RAIZ)
    return cfg


def ruta_proyecto(relativa) -> Path:
    """Convierte una ruta relativa al proyecto en absoluta."""
    p = Path(relativa)
    return p if p.is_absolute() else RAIZ / p


def asegurar_carpeta(ruta) -> Path:
    p = ruta_proyecto(ruta)
    p.mkdir(parents=True, exist_ok=True)
    return p


def marca_tiempo() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def guardar_vector(ruta, vector) -> Path:
    """Guarda un vector de landmarks (float32) en formato .npy."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "wb") as f:          # evita que numpy cambie la extensión
        np.save(f, np.asarray(vector, dtype=np.float32))
    return ruta


def cargar_vector(ruta) -> np.ndarray:
    return np.load(ruta)


def leer_json(ruta, defecto):
    try:
        with open(ruta, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return defecto


def guardar_json(ruta, datos) -> None:
    """Escritura atómica: primero a un .tmp y luego se reemplaza el original."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tmp = ruta.with_suffix(ruta.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(datos, f, ensure_ascii=False, indent=2)
    os.replace(tmp, ruta)


def agregar_fila_csv(ruta, encabezados, fila) -> None:
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    nuevo = not ruta.exists() or ruta.stat().st_size == 0
    with open(ruta, "a", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f)
        if nuevo:
            escritor.writerow(encabezados)
        escritor.writerow(fila)


def leer_csv_dicts(ruta) -> list:
    ruta = Path(ruta)
    if not ruta.exists() or ruta.stat().st_size == 0:
        return []
    with open(ruta, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
