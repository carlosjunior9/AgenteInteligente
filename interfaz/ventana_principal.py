"""Ventana principal: menú con los dos modos."""
import tkinter as tk
from tkinter import ttk

from interfaz.modulo_aprendizaje import ModuloAprendizaje
from interfaz.modulo_traduccion import ModuloTraduccion


class VentanaPrincipal(tk.Tk):
    def __init__(self, cfg: dict, orq):
        super().__init__()
        self.title("Agente Inteligente de Lengua de Señas")
        self.geometry("1020x640")
        self.minsize(900, 600)
        self.cfg = cfg
        self.orq = orq
        self._modulo = None
        self.protocol("WM_DELETE_WINDOW", self._cerrar)

        self._contenedor = tk.Frame(self)
        self._contenedor.pack(fill="both", expand=True)
        self._mostrar_menu()

    def _limpiar(self):
        if self._modulo is not None:
            self._modulo.detener()
        for w in self._contenedor.winfo_children():
            w.destroy()
        self._modulo = None

    def _mostrar_menu(self):
        self._limpiar()
        menu = tk.Frame(self._contenedor)
        menu.place(relx=0.5, rely=0.5, anchor="center")
        tk.Label(menu, text="Lengua de Señas", font=("Helvetica", 26, "bold")).pack(pady=(0, 6))
        tk.Label(menu, text="Elige un modo", font=("Helvetica", 13), fg="#555").pack(pady=(0, 24))
        ttk.Button(menu, text="🎓  Modo aprendizaje", width=28, command=self._abrir_aprendizaje).pack(pady=8, ipady=10)
        ttk.Button(menu, text="🔤  Modo traductor", width=28, command=self._abrir_traduccion).pack(pady=8, ipady=10)
        tk.Label(menu, text="Aprendizaje: enseña señas con la cámara o con fotos.\n"
                            "Traductor: muestra una seña y te digo qué significa.",
                 font=("Helvetica", 10), fg="#777", justify="center").pack(pady=(24, 0))

    def _abrir_aprendizaje(self):
        self._limpiar()
        self._modulo = ModuloAprendizaje(self._contenedor, self.orq, self._mostrar_menu)
        self._modulo.pack(fill="both", expand=True)

    def _abrir_traduccion(self):
        self._limpiar()
        self._modulo = ModuloTraduccion(self._contenedor, self.orq, self._mostrar_menu)
        self._modulo.pack(fill="both", expand=True)

    def _cerrar(self):
        self._limpiar()
        self.orq.cerrar()
        self.destroy()
