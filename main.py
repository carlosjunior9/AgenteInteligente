"""Punto de entrada: python main.py"""
import sys

from utils.io_helpers import cargar_config


def main() -> None:
    cfg = cargar_config()
    try:
        import tkinter  # noqa: F401
    except ImportError:
        print("Falta tkinter. En Linux: sudo apt install python3-tk (en Windows/macOS viene con Python).")
        sys.exit(1)

    from agentes.agente_orquestador import OrquestadorSenas
    from interfaz.ventana_principal import VentanaPrincipal

    try:
        orq = OrquestadorSenas(cfg)
    except RuntimeError as e:
        print(f"Error al iniciar: {e}")
        sys.exit(1)

    VentanaPrincipal(cfg, orq).mainloop()


if __name__ == "__main__":
    main()
