"""Constantes compartidas por todo el proyecto."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent

NUM_LANDMARKS = 21           # puntos de la mano que entrega MediaPipe
DIMENSIONES_POR_PUNTO = 2    # versión 2D: (x, y)
MAX_MANOS = 2
DIM_MANO = NUM_LANDMARKS * DIMENSIONES_POR_PUNTO   # 42
DIM_VECTOR = MAX_MANOS * DIM_MANO                  # 84

INDICE_MUNECA = 0            # landmark 0
INDICE_BASE_MEDIO = 9        # landmark 9: base del dedo medio

# Conexiones de la mano para dibujar el esqueleto (pares de índices).
CONEXIONES_MANO = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # pulgar
    (0, 5), (5, 6), (6, 7), (7, 8),          # índice
    (5, 9), (9, 10), (10, 11), (11, 12),     # medio
    (9, 13), (13, 14), (14, 15), (15, 16),   # anular
    (13, 17), (17, 18), (18, 19), (19, 20),  # meñique
    (0, 17),
]
