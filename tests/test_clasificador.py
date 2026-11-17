import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agentes.agente_aprendizaje import GestorProgreso, SesionPractica, registrar_muestra  # noqa: E402
from agentes.agente_clasificador import ClasificadorSenas, Suavizador, reentrenar  # noqa: E402
from utils.constantes import DIM_VECTOR  # noqa: E402
from utils.io_helpers import cargar_config  # noqa: E402

CFG = {"capas_ocultas": [32], "max_iter": 400, "copias_aumento": 2, "ruido_aumento": 0.01,
       "umbral_confianza": 0.75}


def _datos_sinteticos(n_por_clase=15, seed=0):
    """Tres 'señas' como nubes de puntos alrededor de centros distintos."""
    rng = np.random.default_rng(seed)
    centros = {c: rng.uniform(-1, 1, DIM_VECTOR) for c in ["hola", "casa", "gracias"]}
    X, y = [], []
    for etiqueta, centro in centros.items():
        for _ in range(n_por_clase):
            X.append(centro + rng.normal(0, 0.02, DIM_VECTOR))
            y.append(etiqueta)
    return np.array(X), np.array(y), centros


def test_entrena_y_predice_las_senas_sinteticas():
    X, y, centros = _datos_sinteticos()
    clf = ClasificadorSenas(cfg=CFG)
    metricas = clf.entrenar(X, y)
    assert metricas["accuracy_val"] is not None and metricas["accuracy_val"] >= 0.9
    assert clf.listo and sorted(clf.clases) == ["casa", "gracias", "hola"]

    probs = clf.probabilidades(centros["casa"])
    assert np.isclose(probs.sum(), 1.0)                 # softmax
    etiqueta, conf = clf.decidir(probs)
    assert etiqueta == "casa" and conf > 0.75


def test_umbral_de_confianza_devuelve_no_reconocido():
    X, y, _ = _datos_sinteticos()
    clf = ClasificadorSenas(cfg=CFG)
    clf.entrenar(X, y)
    etiqueta, _ = clf.decidir(np.array([0.34, 0.33, 0.33]), umbral=0.75)
    assert etiqueta is None


def test_guardar_y_cargar_mantiene_predicciones(tmp_path):
    X, y, centros = _datos_sinteticos()
    clf = ClasificadorSenas(cfg=CFG)
    clf.entrenar(X, y)
    ruta = clf.guardar(tmp_path / "m.pkl")
    otro = ClasificadorSenas(cfg=CFG)
    assert otro.cargar(ruta)
    assert np.allclose(clf.probabilidades(centros["hola"]), otro.probabilidades(centros["hola"]))


def test_necesita_al_menos_dos_senas():
    X, y, _ = _datos_sinteticos()
    with pytest.raises(ValueError):
        ClasificadorSenas(cfg=CFG).entrenar(X[y == "hola"], y[y == "hola"])


def test_suavizado_ema_formula():
    s = Suavizador("ema", alpha=0.3)
    a, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    s.actualizar(a)
    out = s.actualizar(b)
    assert np.allclose(out, 0.3 * b + 0.7 * a)


def test_suavizado_ventana_promedia():
    s = Suavizador("ventana", ventana=2)
    s.actualizar(np.array([1.0, 0.0]))
    assert np.allclose(s.actualizar(np.array([0.0, 1.0])), [0.5, 0.5])


def test_reentrenar_end_to_end_con_carpetas_temporales(tmp_path):
    cfg = cargar_config()
    cfg["clasificador"] = {**cfg["clasificador"], **CFG,
                           "modelo_ruta": str(tmp_path / "modelo.pkl")}
    cfg["rutas"] = {
        "capturas": str(tmp_path / "capturas"),
        "etiquetas_base": str(tmp_path / "no_existe.csv"),
        "reentrenados": str(tmp_path / "versiones"),
        "historial": str(tmp_path / "historial.csv"),
        "feedback": str(tmp_path / "feedback.csv"),
        "progreso": str(tmp_path / "progreso.json"),
    }
    X, y, _ = _datos_sinteticos(n_por_clase=6)
    for vec, etiqueta in zip(X, y):
        registrar_muestra(etiqueta, vec, cfg, origen="prueba")

    resultado = reentrenar(cfg, incluir_base=False, guardar_version=True)
    assert resultado["n_senas"] == 3 and resultado["version"] == 1
    assert (tmp_path / "modelo.pkl").exists()
    assert (tmp_path / "versiones" / "modelo_v001.pkl").exists()
    assert (tmp_path / "historial.csv").read_text().count("\n") == 2  # encabezado + 1 fila


def test_progreso_repeticion_espaciada(tmp_path):
    cfg = cargar_config()
    cfg["rutas"] = {**cfg["rutas"], "progreso": str(tmp_path / "p.json")}
    g = GestorProgreso(cfg)
    eta = cfg["aprendizaje"]["eta"]

    assert np.isclose(g.registrar_resultado("hola", True), eta)            # 0 + η(1-0)
    assert np.isclose(g.registrar_resultado("hola", False), eta * (1 - eta))  # s(1-η)
    assert not g.dominada("hola")


def test_decaimiento_por_dias_sin_practicar(tmp_path):
    cfg = cargar_config()
    cfg["rutas"] = {**cfg["rutas"], "progreso": str(tmp_path / "p.json")}
    g = GestorProgreso(cfg)
    g.datos["senas"]["casa"] = {"score": 0.8, "aciertos": 1, "fallos": 0,
                                "ultima_practica": datetime(2026, 1, 1).isoformat()}
    lam = cfg["aprendizaje"]["lambda_decaimiento"]
    dias = 10
    esperado = 0.8 * np.exp(-lam * dias)
    obtenido = g.score_efectivo("casa", ahora=datetime(2026, 1, 1) + timedelta(days=dias))
    assert np.isclose(obtenido, esperado)


def test_sesion_aprueba_tras_15_frames_consecutivos():
    cfg = cargar_config()
    s = SesionPractica("casa", cfg)
    for _ in range(14):
        assert s.actualizar("casa", 0.9) == "en_curso"
    s.actualizar("hola", 0.9)                  # se rompe la racha
    assert s.consecutivos == 0
    for _ in range(15):
        estado = s.actualizar("casa", 0.85)
    assert estado == "aprobada"
