"""Modo traductor.

- Cámara en vivo: muestra la seña y el sistema dice "Significa: casa".
- Escanear foto: abres una imagen y el sistema traduce la seña que contiene.
- Junto a la traducción hay un botón 🔊 Repetir que dice la palabra en voz alta.
"""
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

from agentes.agente_aprendizaje import registrar_correccion
from interfaz.widgets.camara_widget import (
    CamaraWidget, decodificar_imagen, dibujar_manos, mostrar_imagen,
)


class ModuloTraduccion(tk.Frame):
    def __init__(self, master, orq, volver):
        super().__init__(master, padx=12, pady=12)
        self.orq = orq
        self.cfg = orq.cfg
        self.volver = volver
        self._palabra = None        # última seña reconocida (la que dice el botón Repetir)
        self._ultimo = None         # último Resultado (para corregir)

        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self._construir()
        self._cambiar_modo("camara")

    # ---------- pantalla ----------
    def _construir(self):
        cam_cfg = self.cfg["camara"]
        self.visor = CamaraWidget(self, indice=cam_cfg["indice"], ancho=cam_cfg["ancho"],
                                  alto=cam_cfg["alto"], fps=cam_cfg["fps_objetivo"],
                                  procesar=self._procesar_camara)
        self.visor.grid(row=0, column=0, sticky="n")

        p = tk.Frame(self)
        p.grid(row=0, column=1, sticky="nsew", padx=(16, 0))
        tk.Label(p, text="Modo traductor", font=("Helvetica", 18, "bold")).pack(anchor="w")

        modos = tk.Frame(p)
        modos.pack(fill="x", pady=(10, 0))
        ttk.Button(modos, text="📷 Cámara en vivo", command=lambda: self._cambiar_modo("camara")).pack(side="left", fill="x", expand=True)
        ttk.Button(modos, text="🖼 Escanear foto…", command=lambda: self._cambiar_modo("foto")).pack(side="left", fill="x", expand=True, padx=(6, 0))

        tk.Label(p, text="Significa:", font=("Helvetica", 12)).pack(anchor="w", pady=(18, 0))
        fila = tk.Frame(p)
        fila.pack(fill="x")
        self.lbl_palabra = tk.Label(fila, text="—", font=("Helvetica", 30, "bold"), anchor="w")
        self.lbl_palabra.pack(side="left", fill="x", expand=True)
        self.btn_repetir = ttk.Button(fila, text="🔊 Repetir", command=self._repetir, state="disabled")
        self.btn_repetir.pack(side="right")

        self.lbl_detalle = tk.Label(p, text="", font=("Helvetica", 11), fg="#555",
                                    wraplength=340, justify="left")
        self.lbl_detalle.pack(anchor="w", pady=(4, 0))

        ttk.Separator(p).pack(fill="x", pady=14)
        ttk.Button(p, text="✏ Corregir la traducción", command=self._corregir).pack(fill="x")
        ttk.Button(p, text="⬅ Volver al menú", command=self.volver).pack(fill="x", pady=(6, 0))

    # ---------- modos ----------
    def _cambiar_modo(self, modo: str):
        self._modo = modo
        if modo == "camara":
            if not self.visor.iniciar():
                self._mostrar_detalle("No se pudo abrir la cámara.")
                return
            self._mostrar_detalle("Muestra la seña frente a la cámara.")
        else:
            self.visor.detener()
            self._mostrar_detalle("Elige una foto con una seña. Se traducirá al instante.")
            ruta = filedialog.askopenfilename(
                parent=self, title="Elige una foto",
                filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp"), ("Todos", "*.*")])
            if ruta:
                self._traducir_foto(ruta)
        self._sin_modelo_aviso()

    def _sin_modelo_aviso(self):
        if not self.orq.clasificador.listo:
            self.lbl_palabra.configure(text="Aún no hay señas")
            self._mostrar_detalle("Ve a Modo aprendizaje y enseña al menos 2 señas "
                                  "(ej. «hola» y «casa»), luego entrena el modelo.")

    # ---------- cámara en vivo ----------
    def _procesar_camara(self, frame):
        if not self.orq.clasificador.listo:
            return frame
        res = self.orq.procesar(frame)
        self._ultimo = res
        self._pintar(res)
        return dibujar_manos(frame, res.puntos)

    # ---------- foto ----------
    def _traducir_foto(self, ruta):
        img = decodificar_imagen(ruta)
        if img is None:
            messagebox.showerror("Imagen", "No se pudo abrir la imagen.", parent=self)
            return
        if not self.orq.clasificador.listo:
            self._sin_modelo_aviso()
            return
        res = self.orq.procesar(img, suavizar=False)
        self._ultimo = res
        self._pintar(res)
        mostrar_imagen(self.visor, dibujar_manos(img, res.puntos), 640, 480)

    # ---------- presentación ----------
    def _pintar(self, res):
        if res.manos == 0:
            self._mostrar_detalle("No veo ninguna mano." + (" Mostrando la última seña." if self._palabra else ""))
            return
        if res.etiqueta is None:
            top = ", ".join(f"{k} {v:.0%}" for k, v in res.probabilidades.items())
            self.lbl_palabra.configure(text="No reconocido")
            self._mostrar_detalle(f"Confianza insuficiente. Posibles: {top}")
            return
        self._palabra = res.etiqueta
        self.lbl_palabra.configure(text=res.etiqueta)
        self.btn_repetir.configure(state="normal")
        self._mostrar_detalle(f"Confianza: {res.confianza:.0%}")

    def _mostrar_detalle(self, texto: str):
        self.lbl_detalle.configure(text=texto)

    def _repetir(self):
        if self._palabra:
            self.orq.hablar(self._palabra)

    # ---------- corrección ----------
    def _corregir(self):
        if self._ultimo is None or self._ultimo.vector is None:
            messagebox.showinfo("Corregir", "Primero muestra una seña.", parent=self)
            return
        real = simpledialog.askstring("Corregir", "¿Qué significa realmente esta seña?", parent=self)
        if not real or not real.strip():
            return
        registrar_correccion(self._palabra, real.strip(), self._ultimo.vector, self.cfg)
        self._mostrar_detalle(f"Guardado como «{real.strip()}». Entrena de nuevo en Modo aprendizaje.")

    def detener(self):
        self.visor.detener()
