"""Agente de aprendizaje.

- Guarda en datos_aprendizaje/ las muestras que graba la cámara o que se suben como imagen.
- Registra correcciones (predije mal → etiqueta real) en feedback_log.csv.
- Lleva el progreso por seña con repetición espaciada:
      acierto:  score_{t+1} = score_t + η·(1 - score_t)
      fallo:    score_{t+1} = score_t·(1 - η)          η ≈ 0.2
      decaimiento sin practicar D días:  score ← score·e^(-λD)
      dominada si score ≥ 0.9
- Sesión de práctica: la seña objetivo debe predecirse con conf ≥ 0.8 durante
  15 frames consecutivos.
"""
import math
from datetime import datetime

import cv2
import numpy as np

from utils.io_helpers import (
    agregar_fila_csv, asegurar_carpeta, guardar_json, guardar_vector,
    leer_json, marca_tiempo, ruta_proyecto,
)
from utils.logger import obtener_logger

log = obtener_logger(__name__)

_CARACTERES_NO_VALIDOS = '<>:"/\\|?*'


def nombre_carpeta(etiqueta: str) -> str:
    """Convierte la etiqueta escrita por el usuario en un nombre de carpeta válido."""
    limpio = etiqueta.strip().lower()
    for c in _CARACTERES_NO_VALIDOS:
        limpio = limpio.replace(c, "_")
    return limpio


def registrar_muestra(etiqueta: str, vector, cfg: dict, frame_bgr=None, origen="camara"):
    """Guarda un vector de 84 valores (y opcionalmente una miniatura) en capturas_usuario/<seña>/."""
    nombre = nombre_carpeta(etiqueta)
    if not nombre:
        raise ValueError("La seña necesita un nombre.")
    carpeta = asegurar_carpeta(ruta_proyecto(cfg["rutas"]["capturas"]) / nombre)
    base = f"{origen}_{marca_tiempo()}"
    ruta_npy = guardar_vector(carpeta / f"{base}.npy", vector)

    if frame_bgr is not None and cfg["aprendizaje"].get("guardar_imagenes", True):
        alto, ancho = frame_bgr.shape[:2]
        escala = 320 / ancho
        miniatura = cv2.resize(frame_bgr, (320, int(alto * escala)))
        cv2.imwrite(str(carpeta / f"{base}.jpg"), miniatura)
    return ruta_npy


def registrar_correccion(prediccion, real: str, vector, cfg: dict):
    """Guarda la muestra con su etiqueta real y deja constancia en el log de feedback."""
    ruta = registrar_muestra(real, vector, cfg, origen="correccion")
    agregar_fila_csv(
        ruta_proyecto(cfg["rutas"]["feedback"]),
        ["fecha", "prediccion", "real", "ruta_muestra"],
        [datetime.now().isoformat(timespec="seconds"), prediccion or "", real, str(ruta)],
    )
    return ruta


def contar_muestras(cfg: dict) -> dict:
    """{seña: número de muestras guardadas}."""
    raiz = ruta_proyecto(cfg["rutas"]["capturas"])
    conteos = {}
    if raiz.exists():
        for carpeta in sorted(p for p in raiz.iterdir() if p.is_dir()):
            n = len(list(carpeta.glob("*.npy")))
            if n:
                conteos[carpeta.name] = n
    return conteos


class GestorProgreso:
    """Progreso por seña guardado en progreso_usuario.json."""

    def __init__(self, cfg: dict):
        self.ruta = ruta_proyecto(cfg["rutas"]["progreso"])
        self.eta = cfg["aprendizaje"]["eta"]
        self.umbral_dominada = cfg["aprendizaje"]["umbral_dominada"]
        self.lmbda = cfg["aprendizaje"]["lambda_decaimiento"]
        self.datos = leer_json(self.ruta, {"senas": {}})
        self.datos.setdefault("senas", {})

    def _entrada(self, sena: str) -> dict:
        return self.datos["senas"].setdefault(
            sena, {"score": 0.0, "aciertos": 0, "fallos": 0, "ultima_practica": None})

    def registrar_resultado(self, sena: str, acierto: bool) -> float:
        e = self._entrada(sena)
        s = e["score"]
        e["score"] = s + self.eta * (1 - s) if acierto else s * (1 - self.eta)
        e["aciertos" if acierto else "fallos"] += 1
        e["ultima_practica"] = datetime.now().isoformat(timespec="seconds")
        self.guardar()
        return e["score"]

    def score_efectivo(self, sena: str, ahora: datetime | None = None) -> float:
        """Score con decaimiento por días sin practicar (no modifica el archivo)."""
        e = self.datos["senas"].get(sena)
        if not e:
            return 0.0
        if not e.get("ultima_practica"):
            return e["score"]
        ahora = ahora or datetime.now()
        dias = max(0.0, (ahora - datetime.fromisoformat(e["ultima_practica"])).total_seconds() / 86400)
        return e["score"] * math.exp(-self.lmbda * dias)

    def dominada(self, sena: str) -> bool:
        return self.score_efectivo(sena) >= self.umbral_dominada

    def resumen(self) -> dict:
        return {s: self.score_efectivo(s) for s in self.datos["senas"]}

    def guardar(self) -> None:
        guardar_json(self.ruta, self.datos)


class SesionPractica:
    """Evalúa una práctica frame a frame. Estados: en_curso, aprobada, fallida."""

    def __init__(self, objetivo: str, cfg: dict):
        a = cfg["aprendizaje"]
        self.objetivo = objetivo
        self.umbral_conf = a["umbral_practica"]
        self.frames_requeridos = a["frames_practica"]
        self.max_frames = a["max_frames_practica"]
        self.consecutivos = 0
        self.frames = 0
        self.estado = "en_curso"

    def actualizar(self, etiqueta, confianza: float) -> str:
        if self.estado != "en_curso":
            return self.estado
        self.frames += 1
        if etiqueta == self.objetivo and confianza >= self.umbral_conf:
            self.consecutivos += 1
        else:
            self.consecutivos = 0
        if self.consecutivos >= self.frames_requeridos:
            self.estado = "aprobada"
        elif self.frames >= self.max_frames:
            self.estado = "fallida"
        return self.estado

    @property
    def progreso(self) -> float:
        return min(1.0, self.consecutivos / self.frames_requeridos)
