# Agente Inteligente de Lengua de Señas

Sistema que aprende lengua de señas a partir de tu cámara o de fotos. Tú enseñas
las señas (por ejemplo, "hola"), el sistema guarda lo que ve en `datos_aprendizaje/`,
entrena su modelo y después traduce señas nuevas y las dice en voz alta.

Tiene dos modos:

- **Modo aprendizaje**: grabas una seña con la cámara o subes fotos y escribes
  qué significa. También puedes entrenar el modelo y practicar una seña.
- **Modo traductor**: con la cámara en vivo o con una foto, el sistema muestra
  "Significa: casa". Junto a la traducción hay un botón **🔊 Repetir** que dice la
  palabra en voz alta.

## Instalación

Requiere Python 3.10 a 3.12. MediaPipe puede no tener paquete para versiones más nuevas de Python.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

En Linux, tkinter (la interfaz) se instala aparte: `sudo apt install python3-tk`.
En Windows y macOS viene con Python.

La primera vez que se ejecuta, el sistema descarga el modelo de manos de MediaPipe
(~8 MB) a `modelos/hand_landmarker.task`. Si el equipo no tiene internet, descarga
el archivo desde la URL que aparece en `config.yaml` (`deteccion.modelo_url`) y
guárdalo en esa carpeta.

## Uso

```bash
python main.py
```

1. **Modo aprendizaje** → escribe el significado (ej. `hola`) → **Grabar seña con la cámara**
   (haz la seña y muévela un poco; se guardan muestras cada 150 ms).
   Para fotos: escribe el significado y pulsa **Subir imágenes**. Si no escribes
   nada, te preguntará el significado de cada imagen.
2. Enseña al menos **2 señas distintas** (recomendado: 10–30 muestras por seña).
3. **Entrenar modelo**. Al terminar muestra la precisión de validación cuando hay
   muestras suficientes.
4. **Modo traductor** → cámara en vivo o **Escanear foto**. Si el resultado es
   incorrecto, pulsa **Corregir la traducción** e indica la seña real; la muestra
   se guarda y se usa en el siguiente entrenamiento.

Para reentrenar desde consola:

```bash
python scripts/reentrenar_con_usuario.py   # solo muestras del usuario, guarda versión
python scripts/entrenar_modelo.py          # dataset base + usuario
```

Para convertir un dataset propio (carpeta por seña con fotos o videos):

```bash
python scripts/extraer_landmarks.py ruta/a/mi_dataset
```

## Cómo funciona (agentes)

```
frame → detector → normalizador → clasificador → suavizado → voz + GUI
```

| Agente | Archivo | Qué hace |
|---|---|---|
| Detector de manos | `agentes/agente_detector_manos.py` | MediaPipe Hands: 21 landmarks (x, y) por mano. |
| Normalizador | `agentes/agente_normalizador.py` | Traslación a la muñeca, escala por ‖p₉ − p₀‖ y rotación para alinear con el eje Y. Vector fijo de 84 valores (2 manos). |
| Clasificador | `agentes/agente_clasificador.py` | MLP con ReLU y softmax, entrenado con entropía cruzada. Umbral de confianza 0.75. Suavizado EMA (α = 0.3) o ventana de 5 frames. |
| Aprendizaje | `agentes/agente_aprendizaje.py` | Guarda muestras y correcciones, lleva el progreso por seña (repetición espaciada) y evalúa las prácticas (15 frames seguidos con conf ≥ 0.8). |
| Voz | `agentes/agente_voz.py` | Habla con pyttsx3 (offline) o gTTS (online). Solo habla cuando la seña cambia y la confianza es ≥ 0.8. |
| Orquestador | `agentes/agente_orquestador.py` | Une todo el pipeline. |

Las ecuaciones son las de la lista que enviaste. Hay un detalle: en la lista, la
matriz de rotación con θ = arctan2(Δy, Δx) deja el vector sobre el eje **X**. Para
alinearlo con el eje **Y**, el código usa φ = θ − π/2. Esto lo verifican los tests.

## Datos que se guardan

Todo queda en `datos_aprendizaje/`:

- `capturas_usuario/<seña>/*.npy` — vector de 84 valores por muestra (y una
  miniatura `.jpg` si `guardar_imagenes: true`).
- `feedback/feedback_log.csv` — correcciones (predicción, real, ruta).
- `historial_entrenamiento.csv` — métricas de cada entrenamiento.
- `progreso_usuario.json` — score de dominio por seña.
- `modelos_reentrenados/modelo_vNNN.pkl` — versiones del modelo.

El modelo activo está en `modelos/modelo_estatico.pkl`.

## Estructura

```
AgenteInteligente/
├── main.py                       # punto de entrada (abre la GUI)
├── config.yaml                   # umbrales, cámara, voz, rutas
├── requirements.txt
├── agentes/                      # lógica (los 6 agentes)
├── interfaz/
│   ├── ventana_principal.py      # menú: Aprendizaje / Traductor
│   ├── modulo_aprendizaje.py
│   ├── modulo_traduccion.py
│   └── widgets/camara_widget.py  # cámara en vivo, dibujo de manos, carga de imágenes
├── modelos/                      # modelo activo (se crea al entrenar)
├── datos_aprendizaje/            # muestras, feedback, historial, progreso
├── datasets/                     # dataset base opcional (etiquetas.csv + procesado/)
├── scripts/                      # entrenar, reentrenar, extraer landmarks
├── utils/                        # config, io, constantes, logger
└── tests/                        # pruebas de normalizador, clasificador y progreso
```

Ejecutar las pruebas: `pytest tests`

## Limitaciones

- El sistema aprende solo las señas que tú enseñas. No trae un diccionario de LSA.
  Para reconocer un vocabulario grande necesitas un dataset (ver `extraer_landmarks.py`).
- Usa solo la forma de la mano en 2D. Las señas que cambian de movimiento
  (dinámicas) no se distinguen bien de las estáticas; el modelo es estático.
- Con pocas muestras (menos de 10 por seña) el modelo puede confundir señas
  parecidas. Graba en distintas posiciones e iluminación.
- Si la cámara muestra la imagen en espejo, el orden de dos manos puede variar; el
  sistema ordena las manos por posición de la muñeca, así que fotos y cámara
  coinciden mientras la imagen no se voltee.
- La voz con gTTS necesita internet; pyttsx3 necesita una voz en español instalada
  en el sistema para sonar en español.
#para instalar
py -m venv .venv
.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt