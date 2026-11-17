"""Entrena el modelo activo con el dataset base (datasets/etiquetas.csv) + las muestras del usuario.

Uso:
    python scripts/entrenar_modelo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentes.agente_clasificador import reentrenar  # noqa: E402
from utils.io_helpers import cargar_config  # noqa: E402


def main() -> None:
    cfg = cargar_config()
    res = reentrenar(cfg, incluir_base=True, guardar_version=False)
    print(f"Modelo activo guardado: {res['n_muestras']} muestras, {res['n_senas']} señas")
    print(f"accuracy_val={res['accuracy_val']}  f1_val={res['f1_val']}")


if __name__ == "__main__":
    main()
