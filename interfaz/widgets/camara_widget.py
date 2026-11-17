"""Widgets compartidos por los dos modos: vista de cámara, dibujo de manos y carga de imágenes."""
import tkinter as tk

import cv2
import numpy as np
from PIL import Image, ImageTk

from utils.constantes import CONEXIONES_MANO
from utils.logger import obtener_logger

log = obtener_logger(__name__)


def dibujar_manos(frame_bgr: np.ndarray, lista_puntos) -> np.ndarray:
    """Dibuja el esqueleto de cada mano sobre una COPIA del frame."""
    salida = frame_bgr.copy()
    alto, ancho = salida.shape[:2]
    for puntos in lista_puntos:
        px = [(int(x * ancho), int(y * alto)) for x, y in puntos]
        for a, b in CONEXIONES_MANO:
            cv2.line(salida, px[a], px[b], (0, 255, 180), 2)
        for p in px:
            cv2.circle(salida, p, 4, (255, 255, 255), -1)
    return salida


def mostrar_imagen(label: tk.Label, frame_bgr: np.ndarray, max_ancho=640, max_alto=480):
    """Muestra un frame BGR dentro de un tk.Label, manteniendo la proporción."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    img.thumbnail((max_ancho, max_alto))
    foto = ImageTk.PhotoImage(img)
    label.configure(image=foto, text="")
    label.image = foto            # evita que Python borre la imagen


def decodificar_imagen(ruta) -> np.ndarray | None:
    """Lee una imagen aunque la ruta tenga acentos o caracteres no ASCII (Windows)."""
    datos = np.fromfile(str(ruta), dtype=np.uint8)
    return cv2.imdecode(datos, cv2.IMREAD_COLOR)


class CamaraWidget(tk.Label):
    """Muestra la cámara en vivo (en espejo) y entrega cada frame a `procesar`.

    `procesar(frame_bgr) -> frame_bgr` recibe el frame SIN espejo (igual que una foto)
    y devuelve el frame con dibujos. El espejo se aplica solo para mostrarlo.
    """

    def __init__(self, master, indice=0, ancho=640, alto=480, fps=20, procesar=None):
        super().__init__(master, bg="#111", fg="white", font=("Helvetica", 12))
        self.indice = indice
        self.ancho = ancho
        self.alto = alto
        self.intervalo_ms = max(15, int(1000 / fps))
        self.procesar = procesar
        self.cap = None
        self._activa = False

    @property
    def activa(self) -> bool:
        return self._activa

    def iniciar(self) -> bool:
        if self._activa:
            return True
        self.cap = cv2.VideoCapture(self.indice)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.ancho)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.alto)
        if not self.cap.isOpened():
            self.cap = None
            self.configure(image="", text=f"No se pudo abrir la cámara {self.indice}.\n"
                                          "Revisa que no la use otra app y el índice en config.yaml.")
            return False
        self._activa = True
        self._ciclo()
        return True

    def detener(self) -> None:
        self._activa = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def _ciclo(self) -> None:
        if not self._activa:
            return
        ok, frame = self.cap.read()
        if ok:
            salida = frame
            if self.procesar is not None:
                try:
                    salida = self.procesar(frame)
                except Exception:
                    log.exception("Error procesando un frame")
            mostrar_imagen(self, cv2.flip(salida, 1), self.ancho, self.alto)
        self.after(self.intervalo_ms, self._ciclo)
