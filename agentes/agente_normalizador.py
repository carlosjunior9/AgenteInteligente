"""Agente 2 - Normalizador.

Hace que el vector de la seña no dependa de dónde está la mano, de su tamaño
ni de la inclinación con la que la pone la persona ni de la cámara.

Ecuaciones (según la lista de ecuaciones del proyecto):
  1. Traslación a la muñeca (landmark 0):      p'_i  = p_i - p_0
  2. Escala por tamaño de la mano:              s = ||p_9 - p_0||_2 ,  p''_i = p'_i / s
  3. Rotación para alinear p_9 - p_0 con el eje Y:
        θ = arctan2(Δy, Δx),   φ = θ - π/2,   R = [[cos φ, sin φ], [-sin φ, cos φ]]
        p'''_i = R · p''_i
     (Con φ = θ la matriz de la lista deja el vector sobre el eje X; el eje Y
     se obtiene restando π/2, por eso aquí se usa φ = θ - π/2.)

Vector final: 2 manos × 21 puntos × (x, y) = 84 valores. Si falta una mano,
su bloque queda en ceros.
"""
import numpy as np

from utils.constantes import (
    DIM_MANO, DIM_VECTOR, INDICE_BASE_MEDIO, INDICE_MUNECA, MAX_MANOS, NUM_LANDMARKS,
)

_EPSILON = 1e-6


def normalizar_mano(puntos, rotar: bool = True) -> np.ndarray:
    """Normaliza una mano (21, 2) y devuelve un vector de 42 valores."""
    p = np.asarray(puntos, dtype=np.float64).reshape(NUM_LANDMARKS, 2)

    # 1) Traslación: la muñeca pasa a ser el origen.
    p = p - p[INDICE_MUNECA]

    # 2) Escala: dividir por la distancia muñeca → base del dedo medio.
    delta = p[INDICE_BASE_MEDIO]
    s = np.linalg.norm(delta)
    if s < _EPSILON:
        return np.zeros(DIM_MANO)
    p = p / s

    # 3) Rotación: alinear (p_9 - p_0) con el eje Y.
    if rotar:
        theta = np.arctan2(delta[1], delta[0])
        phi = theta - np.pi / 2
        c, sn = np.cos(phi), np.sin(phi)
        R = np.array([[c, sn], [-sn, c]])
        p = p @ R.T          # equivalente a R · p_i para cada fila

    return p.flatten()       # [x1, y1, x2, y2, ..., x21, y21]


def vector_desde_manos(lista_puntos, rotar: bool = True) -> np.ndarray:
    """Combina hasta MAX_MANOS manos en un vector fijo de 84 valores.

    Las manos se ordenan por la posición x de la muñeca, para que el orden
    sea el mismo en fotos y en cámara.
    """
    vector = np.zeros(DIM_VECTOR)
    if not lista_puntos:
        return vector
    manos = sorted(lista_puntos, key=lambda pts: float(np.asarray(pts)[INDICE_MUNECA, 0]))
    for i, pts in enumerate(manos[:MAX_MANOS]):
        vector[i * DIM_MANO:(i + 1) * DIM_MANO] = normalizar_mano(pts, rotar)
    return vector
