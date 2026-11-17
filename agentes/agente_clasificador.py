"""Agente 3 - Clasificador.

Modelo: MLP (red neuronal de una o más capas ocultas).
  h = ReLU(W1 x + b1)          z = W2 h + b2
  ŷ_k = exp(z_k) / Σ_j exp(z_j)            (softmax)
  L = -Σ_k y_k log(ŷ_k)                    (entropía cruzada)
  predicción = argmax_k ŷ_k ;  conf = max_k ŷ_k ; si conf < umbral → "no reconocido"

También contiene:
  - Suavizador temporal (filtro exponencial o ventana deslizante).
  - El entrenamiento a partir de las muestras del usuario y del dataset base.
"""
import warnings
from datetime import datetime
from collections import deque
from pathlib import Path

import joblib
import numpy as np

from utils.constantes import DIM_VECTOR, RAIZ
from utils.io_helpers import (
    agregar_fila_csv, cargar_vector, leer_csv_dicts, ruta_proyecto,
)
from utils.logger import obtener_logger

log = obtener_logger(__name__)
SEMILLA = 42


class ClasificadorSenas:
    def __init__(self, ruta_modelo=None, cfg: dict | None = None):
        self.cfg = cfg or {}
        self.ruta = Path(ruta_modelo) if ruta_modelo else RAIZ / "modelos" / "modelo_estatico.pkl"
        self.modelo = None
        self.clases: list[str] = []
        self.metadatos: dict = {}

    @property
    def listo(self) -> bool:
        return self.modelo is not None and len(self.clases) >= 2

    # ---------- persistencia ----------
    def cargar(self, ruta=None) -> bool:
        ruta = Path(ruta) if ruta else self.ruta
        if not ruta.exists() or ruta.stat().st_size == 0:
            self.modelo, self.clases, self.metadatos = None, [], {}
            return False
        datos = joblib.load(ruta)
        self.modelo = datos["modelo"]
        self.clases = list(datos["clases"])
        self.metadatos = datos.get("metadatos", {})
        return True

    def guardar(self, ruta=None) -> Path:
        ruta = Path(ruta) if ruta else self.ruta
        ruta.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"modelo": self.modelo, "clases": self.clases,
                     "metadatos": self.metadatos}, ruta)
        return ruta

    # ---------- entrenamiento ----------
    def _crear_modelo(self):
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(
            hidden_layer_sizes=tuple(self.cfg.get("capas_ocultas", [128, 64])),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            max_iter=self.cfg.get("max_iter", 800),
            random_state=SEMILLA,
        )

    def _aumentar(self, X, y):
        """Copias con ruido gaussiano: compensa tener pocas muestras por seña."""
        rng = np.random.default_rng(SEMILLA)
        copias = self.cfg.get("copias_aumento", 4)
        sigma = self.cfg.get("ruido_aumento", 0.01)
        Xs, ys = [X], [y]
        for _ in range(copias):
            Xs.append(X + rng.normal(0.0, sigma, X.shape))
            ys.append(y)
        return np.vstack(Xs), np.concatenate(ys)

    def entrenar(self, X, y) -> dict:
        from sklearn.metrics import accuracy_score, f1_score
        from sklearn.model_selection import train_test_split

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if X.ndim != 2 or X.shape[1] != DIM_VECTOR:
            raise ValueError(f"Se esperaban vectores de {DIM_VECTOR} valores.")
        clases, conteos = np.unique(y, return_counts=True)
        if len(clases) < 2:
            raise ValueError("Se necesitan al menos 2 señas distintas para entrenar.")

        metricas = {"accuracy_val": None, "f1_val": None}
        n_val = int(np.ceil(0.2 * len(y)))
        if len(y) >= 10 and conteos.min() >= 2 and n_val >= len(clases):
            Xtr, Xva, ytr, yva = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=SEMILLA)
            m = self._crear_modelo()
            Xa, ya = self._aumentar(Xtr, ytr)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(Xa, ya)
            pred = m.predict(Xva)
            metricas = {
                "accuracy_val": float(accuracy_score(yva, pred)),
                "f1_val": float(f1_score(yva, pred, average="macro", zero_division=0)),
            }

        # Modelo final: se entrena con todas las muestras.
        Xa, ya = self._aumentar(X, y)
        m = self._crear_modelo()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(Xa, ya)
        self.modelo = m
        self.clases = [str(c) for c in m.classes_]
        self.metadatos = {
            "n_muestras": int(len(y)),
            "n_senas": len(self.clases),
            "clases": self.clases,
            "fecha": datetime.now().isoformat(timespec="seconds"),
            **metricas,
        }
        return dict(self.metadatos)

    # ---------- inferencia ----------
    def probabilidades(self, vector) -> np.ndarray:
        """Vector de probabilidades (softmax) alineado con self.clases."""
        if not self.listo:
            raise RuntimeError("El modelo no está entrenado.")
        v = np.asarray(vector, dtype=np.float64).reshape(1, -1)
        return self.modelo.predict_proba(v)[0]

    def decidir(self, probs, umbral: float | None = None):
        """Aplica argmax y umbral. Devuelve (etiqueta o None, confianza)."""
        umbral = self.cfg.get("umbral_confianza", 0.75) if umbral is None else umbral
        i = int(np.argmax(probs))
        conf = float(probs[i])
        return (self.clases[i] if conf >= umbral else None), conf


class Suavizador:
    """Suavizado temporal de las probabilidades (evita que la predicción parpadee)."""

    def __init__(self, metodo: str = "ema", alpha: float = 0.3, ventana: int = 5):
        self.metodo = metodo
        self.alpha = alpha
        self._estado = None
        self._ventana = deque(maxlen=ventana)

    def actualizar(self, probs) -> np.ndarray:
        probs = np.asarray(probs, dtype=np.float64)
        if self.metodo == "ventana":
            self._ventana.append(probs)
            return np.mean(self._ventana, axis=0)
        # Filtro exponencial: ŷ_t = α·y_t + (1-α)·ŷ_{t-1}
        if self._estado is None or self._estado.shape != probs.shape:
            self._estado = probs
        else:
            self._estado = self.alpha * probs + (1 - self.alpha) * self._estado
        return self._estado

    def reset(self) -> None:
        self._estado = None
        self._ventana.clear()


# ---------- datos para entrenar ----------

def cargar_datos_usuario(cfg: dict):
    """Lee datos_aprendizaje/capturas_usuario/<seña>/*.npy."""
    raiz = ruta_proyecto(cfg["rutas"]["capturas"])
    X, y = [], []
    if raiz.exists():
        for carpeta in sorted(p for p in raiz.iterdir() if p.is_dir()):
            for archivo in sorted(carpeta.glob("*.npy")):
                vec = cargar_vector(archivo)
                if vec.shape == (DIM_VECTOR,):
                    X.append(vec)
                    y.append(carpeta.name)
    return np.array(X).reshape(-1, DIM_VECTOR), np.array(y)


def cargar_datos_base(cfg: dict):
    """Lee datasets/etiquetas.csv (columnas: label, path_landmarks), si existe."""
    X, y = [], []
    for fila in leer_csv_dicts(ruta_proyecto(cfg["rutas"]["etiquetas_base"])):
        ruta = ruta_proyecto(fila["path_landmarks"])
        if not ruta.exists():
            continue
        datos = cargar_vector(ruta)
        datos = datos.reshape(-1, DIM_VECTOR) if datos.ndim == 2 else datos.reshape(1, -1)
        for vec in datos:
            if vec.shape == (DIM_VECTOR,):
                X.append(vec)
                y.append(fila["label"])
    return np.array(X).reshape(-1, DIM_VECTOR), np.array(y)


def reentrenar(cfg: dict, incluir_base: bool = False, guardar_version: bool = True) -> dict:
    """Entrena con las muestras del usuario (y opcionalmente el dataset base).

    Guarda el modelo activo en modelos/ y, si guardar_version=True, una copia
    numerada en datos_aprendizaje/modelos_reentrenados/. Registra métricas en
    historial_entrenamiento.csv.
    """
    partes = [cargar_datos_usuario(cfg)]
    if incluir_base:
        partes.append(cargar_datos_base(cfg))
    X = np.vstack([p[0] for p in partes if len(p[0])])
    y = np.concatenate([p[1] for p in partes if len(p[1])])
    if len(y) == 0:
        raise ValueError("No hay muestras. Enseña al menos dos señas en Modo aprendizaje.")

    clf = ClasificadorSenas(RAIZ / cfg["clasificador"]["modelo_ruta"], cfg["clasificador"])
    metricas = clf.entrenar(X, y)
    clf.guardar()

    version = None
    if guardar_version:
        carpeta = ruta_proyecto(cfg["rutas"]["reentrenados"])
        carpeta.mkdir(parents=True, exist_ok=True)
        existentes = list(carpeta.glob("modelo_v*.pkl"))
        version = len(existentes) + 1
        clf.guardar(carpeta / f"modelo_v{version:03d}.pkl")

    agregar_fila_csv(
        ruta_proyecto(cfg["rutas"]["historial"]),
        ["fecha", "version", "fuente", "n_muestras", "n_senas",
         "accuracy_val", "f1_val"],
        [clf.metadatos["fecha"], version or "activo",
         "base+usuario" if incluir_base else "usuario",
         clf.metadatos["n_muestras"], clf.metadatos["n_senas"],
         _fmt(metricas["accuracy_val"]), _fmt(metricas["f1_val"])],
    )
    log.info("Modelo entrenado: %d muestras, %d señas", len(y), len(clf.clases))
    return {**clf.metadatos, "version": version}


def _fmt(valor):
    return "" if valor is None else f"{valor:.4f}"
