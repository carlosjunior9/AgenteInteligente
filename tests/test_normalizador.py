import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agentes.agente_normalizador import normalizar_mano, vector_desde_manos  # noqa: E402


def _mano_aleatoria(seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.2, 0.8, size=(21, 2))


def _rotar(puntos, angulo):
    c, s = np.cos(angulo), np.sin(angulo)
    R = np.array([[c, -s], [s, c]])
    return puntos @ R.T


def test_dimension_del_vector():
    assert normalizar_mano(_mano_aleatoria()).shape == (42,)
    assert vector_desde_manos([_mano_aleatoria()]).shape == (84,)


def test_invariante_a_traslacion():
    p = _mano_aleatoria(1)
    assert np.allclose(normalizar_mano(p), normalizar_mano(p + np.array([5.0, -3.0])))


def test_invariante_a_escala():
    p = _mano_aleatoria(2)
    assert np.allclose(normalizar_mano(p), normalizar_mano(p * 3.7))


def test_invariante_a_rotacion():
    p = _mano_aleatoria(3)
    assert np.allclose(normalizar_mano(p), normalizar_mano(_rotar(p, 0.9)), atol=1e-9)


def test_base_del_dedo_medio_queda_sobre_eje_y():
    p = _mano_aleatoria(4)
    norm = normalizar_mano(p).reshape(21, 2)
    # Landmark 0 es el origen; landmark 9 queda en (0, 1) tras la escala y la rotación.
    assert np.allclose(norm[0], [0, 0])
    assert np.allclose(norm[9], [0, 1], atol=1e-9)


def test_sin_rotacion_no_alinea_pero_mantiene_escala():
    p = _mano_aleatoria(5)
    sin_rot = normalizar_mano(p, rotar=False).reshape(21, 2)
    assert np.isclose(np.linalg.norm(sin_rot[9]), 1.0)


def test_mano_faltante_da_ceros_en_su_bloque():
    v = vector_desde_manos([_mano_aleatoria(6)])
    assert np.all(v[42:] == 0)
    assert np.any(v[:42] != 0)


def test_sin_manos_devuelve_vector_cero():
    assert np.all(vector_desde_manos([]) == 0)


def test_orden_de_manos_por_posicion_x():
    izq = _mano_aleatoria(7)
    der = _mano_aleatoria(8) + np.array([0.5, 0.0])   # muñeca más a la derecha
    v1 = vector_desde_manos([izq, der])
    v2 = vector_desde_manos([der, izq])
    assert np.allclose(v1, v2)
    assert np.allclose(v1[:42], normalizar_mano(izq))
