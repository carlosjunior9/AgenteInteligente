"""Agente 1 - Detector de manos.

Usa MediaPipe Hands (API Tasks). Por cada mano entrega 21 landmarks (x, y)
normalizados en [0, 1]. Las manos se ordenan de izquierda a derecha según la
posición de la muñeca, así una seña de una mano siempre ocupa la misma ranura.
"""
import urllib.request
from pathlib import Path

import numpy as np

from utils.constantes import RAIZ
from utils.logger import obtener_logger

log = obtener_logger(__name__)


def asegurar_modelo(cfg_det: dict) -> Path:
    """Descarga el modelo hand_landmarker.task la primera vez (~8 MB)."""
    ruta = RAIZ / cfg_det["modelo_ruta"]
    if ruta.exists() and ruta.stat().st_size > 0:
        return ruta
    ruta.parent.mkdir(parents=True, exist_ok=True)
    url = cfg_det["modelo_url"]
    log.info("Descargando modelo de manos de MediaPipe (solo la primera vez)...")
    try:
        urllib.request.urlretrieve(url, ruta)
    except Exception as e:
        if ruta.exists():
            ruta.unlink()
        raise RuntimeError(
            "No se pudo descargar el modelo de manos.\n"
            f"Descárgalo manualmente desde:\n{url}\n"
            f"y guárdalo como:\n{ruta}"
        ) from e
    return ruta


class DetectorManos:
    def __init__(self, cfg_det: dict):
        try:
            import cv2
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
        except ImportError as e:
            raise RuntimeError(
                "Faltan dependencias. Instala con: pip install -r requirements.txt"
            ) from e

        self._cv2 = cv2
        self._mp = mp
        ruta_modelo = asegurar_modelo(cfg_det)
        opciones = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(ruta_modelo)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=cfg_det["max_manos"],
            min_hand_detection_confidence=cfg_det["confianza_deteccion"],
            min_hand_presence_confidence=cfg_det["confianza_presencia"],
            min_tracking_confidence=cfg_det["confianza_seguimiento"],
        )
        self._landmarker = vision.HandLandmarker.create_from_options(opciones)

    def detectar(self, frame_bgr) -> list:
        """Devuelve una lista de dicts: {"puntos": array (21, 2), "x_muneca": float}.
        Ordenada de izquierda a derecha."""
        rgb = self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2RGB)
        imagen = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb),
        )
        resultado = self._landmarker.detect(imagen)
        manos = []
        for landmarks in resultado.hand_landmarks:
            puntos = np.array([[p.x, p.y] for p in landmarks], dtype=np.float64)
            manos.append({"puntos": puntos, "x_muneca": float(puntos[0, 0])})
        manos.sort(key=lambda m: m["x_muneca"])
        return manos

    def cerrar(self) -> None:
        self._landmarker.close()
