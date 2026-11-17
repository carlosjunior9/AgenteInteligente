"""Agente orquestador.

Pipeline:
  frame → detector → normalizador → clasificador → suavizado → voz + GUI

Regla de voz (ecuación de la lista):
  hablar = (seña_t ≠ seña_{t-1}) ∧ (conf_t ≥ 0.8)
  Así "casa" no se repite 30 veces por segundo: solo habla cuando cambia.
"""
from dataclasses import dataclass, field

import numpy as np

from agentes.agente_clasificador import ClasificadorSenas, Suavizador
from agentes.agente_detector_manos import DetectorManos
from agentes.agente_normalizador import vector_desde_manos
from agentes.agente_voz import AgenteVoz
from utils.constantes import RAIZ
from utils.io_helpers import cargar_config
from utils.logger import obtener_logger

log = obtener_logger(__name__)


@dataclass
class Resultado:
    manos: int = 0
    etiqueta: str | None = None          # None = sin seña / no reconocido
    confianza: float = 0.0
    probabilidades: dict = field(default_factory=dict)
    puntos: list = field(default_factory=list)   # landmarks (21, 2) de cada mano
    vector: np.ndarray | None = None              # vector normalizado (84)
    hablado: bool = False                         # True si la voz dijo algo en este frame


class OrquestadorSenas:
    def __init__(self, cfg: dict | None = None, con_voz: bool = True):
        self.cfg = cfg or cargar_config()
        self.detector = DetectorManos(self.cfg["deteccion"])
        self.clasificador = ClasificadorSenas(RAIZ / self.cfg["clasificador"]["modelo_ruta"],
                                              self.cfg["clasificador"])
        self.clasificador.cargar()
        s = self.cfg["suavizado"]
        self.suavizador = Suavizador(s["metodo"], s["alpha"], s["ventana"])
        self.voz = AgenteVoz(self.cfg["voz"]) if con_voz else None
        self._sena_estable_previa = None

    # ---------- utilidades ----------
    def recargar_modelo(self) -> bool:
        self.suavizador.reset()
        self._sena_estable_previa = None
        return self.clasificador.cargar()

    def extraer(self, frame_bgr):
        """Solo detección y normalización (sin clasificar). Devuelve (vector|None, puntos)."""
        manos = self.detector.detectar(frame_bgr)
        if not manos:
            return None, []
        puntos = [m["puntos"] for m in manos]
        return vector_desde_manos(puntos, self.cfg["normalizacion"]["rotar"]), puntos

    # ---------- pipeline completo ----------
    def procesar(self, frame_bgr, suavizar: bool = True) -> Resultado:
        vector, puntos = self.extraer(frame_bgr)
        if vector is None:
            self.suavizador.reset()
            self._sena_estable_previa = None
            return Resultado()

        res = Resultado(manos=len(puntos), puntos=puntos, vector=vector)
        if not self.clasificador.listo:
            return res

        probs = self.clasificador.probabilidades(vector)
        if suavizar:
            probs = self.suavizador.actualizar(probs)

        etiqueta, conf = self.clasificador.decidir(probs)
        orden = np.argsort(probs)[::-1][:3]
        res.etiqueta = etiqueta
        res.confianza = conf
        res.probabilidades = {self.clasificador.clases[i]: float(probs[i]) for i in orden}

        # Regla de voz: solo cuando la seña estable cambia y la confianza es alta.
        umbral_habla = self.cfg["voz"].get("umbral_habla", 0.8)
        sena_estable = etiqueta if conf >= umbral_habla else None
        if (self.voz and self.voz.activa and sena_estable
                and sena_estable != self._sena_estable_previa):
            res.hablado = self.voz.hablar(sena_estable)
        self._sena_estable_previa = sena_estable
        return res

    def hablar(self, texto: str) -> bool:
        """Para el botón 'Repetir'. Funciona aunque la voz automática esté desactivada."""
        if self.voz is None:
            self.voz = AgenteVoz(self.cfg["voz"])
        return self.voz.hablar(texto)

    def cerrar(self) -> None:
        self.detector.cerrar()
