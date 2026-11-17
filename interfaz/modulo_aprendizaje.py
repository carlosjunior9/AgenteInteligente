"""Modo aprendizaje.

1. Escribes el significado (ej. "hola") y grabas la seña con la cámara, o subes
   imágenes de señas y indicas qué significan.
2. Los agentes extraen los landmarks y los guardan en datos_aprendizaje/.
3. Entrenas el modelo y, opcionalmente, practicas una seña aprendida.
"""
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

from agentes.agente_aprendizaje import (
    GestorProgreso, SesionPractica, contar_muestras, registrar_muestra,
)
from agentes.agente_clasificador import reentrenar
from interfaz.widgets.camara_widget import CamaraWidget, decodificar_imagen, dibujar_manos
from utils.io_helpers import leer_csv_dicts, ruta_proyecto

INTERVALO_CAPTURA_S = 0.15   # una muestra cada 150 ms → variedad sin repetir el mismo frame
MAX_FRAMES_CAPTURA = 400     # tope de seguridad si no aparece ninguna mano


class ModuloAprendizaje(tk.Frame):
    def __init__(self, master, orq, volver):
        super().__init__(master, padx=12, pady=12)
        self.orq = orq
        self.cfg = orq.cfg
        self.volver = volver
        self.gestor = GestorProgreso(self.cfg)
        self._captura = None        # dict con el estado de una grabación
        self._practica = None       # (SesionPractica, seña)
        self._senas = []            # nombres en el mismo orden que la lista
        self._cola = queue.Queue()  # resultados de entrenamiento (hilo → GUI)
        self._cerrado = False

        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self._construir_camara()
        self._construir_panel()
        self._refrescar()
        self.cam.iniciar()

    # ---------- construcción de la pantalla ----------
    def _construir_camara(self):
        cam_cfg = self.cfg["camara"]
        self.cam = CamaraWidget(self, indice=cam_cfg["indice"], ancho=cam_cfg["ancho"],
                                alto=cam_cfg["alto"], fps=cam_cfg["fps_objetivo"],
                                procesar=self._al_frame)
        self.cam.grid(row=0, column=0, sticky="n")

    def _construir_panel(self):
        p = tk.Frame(self)
        p.grid(row=0, column=1, sticky="nsew", padx=(16, 0))

        tk.Label(p, text="Modo aprendizaje", font=("Helvetica", 18, "bold")).pack(anchor="w")
        tk.Label(p, text="Significado de la seña (ej. hola):", font=("Helvetica", 11)).pack(anchor="w", pady=(10, 2))
        self.var_etiqueta = tk.StringVar()
        tk.Entry(p, textvariable=self.var_etiqueta, font=("Helvetica", 14)).pack(fill="x")

        ttk.Button(p, text="🎥 Grabar seña con la cámara", command=self._iniciar_captura).pack(fill="x", pady=(10, 4))
        ttk.Button(p, text="🖼 Subir imágenes de señas…", command=self._subir_imagenes).pack(fill="x")

        self.progreso = ttk.Progressbar(p, maximum=100)
        self.progreso.pack(fill="x", pady=(10, 2))
        self.lbl_estado = tk.Label(p, text="", wraplength=340, justify="left", font=("Helvetica", 10),
                                   fg="#333")
        self.lbl_estado.pack(anchor="w")

        ttk.Separator(p).pack(fill="x", pady=10)
        tk.Label(p, text="Señas aprendidas", font=("Helvetica", 12, "bold")).pack(anchor="w")
        self.lista = tk.Listbox(p, height=8, font=("Helvetica", 11), activestyle="none")
        self.lista.pack(fill="x", pady=4)
        self.lbl_modelo = tk.Label(p, text="", font=("Helvetica", 10), fg="#555", wraplength=340, justify="left")
        self.lbl_modelo.pack(anchor="w")
        self.btn_entrenar = ttk.Button(p, text="🧠 Entrenar modelo", command=self._entrenar)
        self.btn_entrenar.pack(fill="x", pady=(6, 2))

        ttk.Separator(p).pack(fill="x", pady=10)
        tk.Label(p, text="Práctica", font=("Helvetica", 12, "bold")).pack(anchor="w")
        tk.Label(p, text="Selecciona una seña de la lista y hazla frente a la cámara "
                         "durante 15 frames seguidos.", wraplength=340, justify="left",
                 font=("Helvetica", 10), fg="#555").pack(anchor="w")
        ttk.Button(p, text="▶ Practicar la seña seleccionada", command=self._iniciar_practica).pack(fill="x", pady=4)

        ttk.Separator(p).pack(fill="x", pady=10)
        ttk.Button(p, text="⬅ Volver al menú", command=self.volver).pack(fill="x")

    # ---------- ciclo de frames (lo llama CamaraWidget) ----------
    def _al_frame(self, frame):
        if self._practica is not None:
            return self._frame_practica(frame)

        vector, puntos = self.orq.extraer(frame)
        if self._captura is not None:
            if vector is not None:
                self._captura["sin_mano"] = 0
                self._guardar_muestra_captura(frame, vector)
            else:
                self._captura["sin_mano"] += 1
                if self._captura["sin_mano"] > MAX_FRAMES_CAPTURA:
                    self._terminar_captura(motivo="No se detectaron manos. Intenta de nuevo con buena luz.")
        return dibujar_manos(frame, puntos)

    def _frame_practica(self, frame):
        res = self.orq.procesar(frame)
        sesion, sena = self._practica
        estado = sesion.actualizar(res.etiqueta, res.confianza)
        self.progreso["value"] = sesion.progreso * 100
        if estado == "en_curso":
            self.lbl_estado.configure(text=f"Practicando «{sena}»… racha: {sesion.consecutivos}/"
                                           f"{sesion.frames_requeridos}")
        else:
            self._terminar_practica(sesion, sena, aprobada=(estado == "aprobada"))
        return dibujar_manos(frame, res.puntos)

    # ---------- grabar con cámara ----------
    def _iniciar_captura(self):
        etiqueta = self.var_etiqueta.get().strip()
        if not etiqueta:
            messagebox.showinfo("Falta el significado", "Escribe primero qué significa la seña (ej. hola).", parent=self)
            return
        if not self.cam.activa:
            messagebox.showerror("Cámara", "La cámara no está activa.", parent=self)
            return
        n = self.cfg["aprendizaje"]["muestras_por_captura"]
        self._captura = {"etiqueta": etiqueta, "meta": n, "guardadas": 0,
                         "ultimo": 0.0, "sin_mano": 0}
        self.progreso.configure(maximum=n, value=0)
        self._estado(f"Grabando «{etiqueta}»… haz la seña y muévela un poco para tener variedad.")

    def _guardar_muestra_captura(self, frame, vector):
        ahora = time.monotonic()
        c = self._captura
        if ahora - c["ultimo"] < INTERVALO_CAPTURA_S:
            return
        c["ultimo"] = ahora
        registrar_muestra(c["etiqueta"], vector, self.cfg, frame_bgr=frame, origen="camara")
        c["guardadas"] += 1
        self.progreso["value"] = c["guardadas"]
        if c["guardadas"] >= c["meta"]:
            self._terminar_captura(motivo=f"Listo: {c['guardadas']} muestras de «{c['etiqueta']}» guardadas.")

    def _terminar_captura(self, motivo: str):
        self._captura = None
        self._estado(motivo)
        self._refrescar()

    # ---------- subir imágenes ----------
    def _subir_imagenes(self):
        rutas = filedialog.askopenfilenames(
            parent=self, title="Elige imágenes de señas",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp"), ("Todos", "*.*")])
        if not rutas:
            return
        etiqueta_comun = self.var_etiqueta.get().strip()
        guardadas, sin_mano = 0, []
        for ruta in rutas:
            etiqueta = etiqueta_comun or simpledialog.askstring(
                "Significado", f"¿Qué significa la seña de la imagen:\n{Path(ruta).name}?", parent=self)
            if not etiqueta:
                continue
            img = decodificar_imagen(ruta)
            if img is None:
                sin_mano.append(Path(ruta).name)
                continue
            vector, _ = self.orq.extraer(img)
            if vector is None:
                sin_mano.append(Path(ruta).name)
                continue
            registrar_muestra(etiqueta, vector, self.cfg, frame_bgr=img, origen="imagen")
            guardadas += 1
        msg = f"Se guardaron {guardadas} imagen(es) de señas."
        if sin_mano:
            msg += f"\nSin mano detectada: {', '.join(sin_mano)}"
        self._estado(msg)
        self._refrescar()

    # ---------- entrenamiento en segundo plano ----------
    def _entrenar(self):
        hay_base = bool(leer_csv_dicts(ruta_proyecto(self.cfg["rutas"]["etiquetas_base"])))
        self.btn_entrenar.configure(state="disabled")
        self._estado("Entrenando el modelo… (unos segundos)")

        def trabajo():
            try:
                res = reentrenar(self.cfg, incluir_base=hay_base, guardar_version=True)
                self._cola.put(("ok", res))
            except Exception as e:
                self._cola.put(("error", str(e)))

        threading.Thread(target=trabajo, daemon=True).start()
        self.after(150, self._revisar_entrenamiento)

    def _revisar_entrenamiento(self):
        if self._cerrado:
            return
        try:
            estado, dato = self._cola.get_nowait()
        except queue.Empty:
            self.after(150, self._revisar_entrenamiento)
            return
        self.btn_entrenar.configure(state="normal")
        if estado == "error":
            self._estado(f"No se pudo entrenar: {dato}")
            messagebox.showwarning("Entrenamiento", dato, parent=self)
            return
        self.orq.recargar_modelo()
        acc = dato.get("accuracy_val")
        precision = f"Precisión de validación: {acc:.0%}" if acc is not None else \
            "Sin validación (pocas muestras por seña; agrega más para medirla)."
        self._estado(f"Modelo v{dato['version']} listo: {dato['n_senas']} señas, "
                     f"{dato['n_muestras']} muestras.\n{precision}")
        self._refrescar()

    # ---------- práctica ----------
    def _iniciar_practica(self):
        seleccion = self.lista.curselection()
        if not seleccion:
            messagebox.showinfo("Práctica", "Selecciona primero una seña de la lista.", parent=self)
            return
        if not self.orq.clasificador.listo:
            messagebox.showinfo("Práctica", "Entrena el modelo antes de practicar.", parent=self)
            return
        sena = self._senas[seleccion[0]]
        self._captura = None
        self._practica = (SesionPractica(sena, self.cfg), sena)
        self.progreso.configure(maximum=100, value=0)
        self._estado(f"Haz la seña «{sena}» frente a la cámara.")

    def _terminar_practica(self, sesion, sena, aprobada: bool):
        score = self.gestor.registrar_resultado(sena, acierto=aprobada)
        self._practica = None
        self.progreso["value"] = 100 if aprobada else 0
        if aprobada:
            self._estado(f"¡Bien! «{sena}» practicada. Dominio: {score:.0%}")
        else:
            self._estado(f"No se logró «{sena}» a tiempo. Dominio: {score:.0%}. Inténtalo otra vez.")
        self._refrescar()

    # ---------- listado y estado ----------
    def _refrescar(self):
        conteos = contar_muestras(self.cfg)
        self._senas = sorted(conteos)
        self.lista.delete(0, tk.END)
        for sena in self._senas:
            dominio = self.gestor.score_efectivo(sena)
            self.lista.insert(tk.END, f"{sena}  ·  {conteos[sena]} muestras  ·  dominio {dominio:.0%}")
        if self.orq.clasificador.listo:
            n = self.orq.clasificador.metadatos.get("n_senas", len(self.orq.clasificador.clases))
            self.lbl_modelo.configure(text=f"Modelo activo: {n} señas reconocibles.")
        else:
            self.lbl_modelo.configure(text="Aún no hay modelo. Enseña al menos 2 señas y entrena.")

    def _estado(self, texto: str):
        self.lbl_estado.configure(text=texto)

    def detener(self):
        self._cerrado = True
        self._captura = None
        self._practica = None
        self.cam.detener()
