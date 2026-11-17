proyecto_senas/
│
├── main.py                          # Punto de entrada (lanza la GUI)
├── requirements.txt
├── config.yaml                      # Configuración global (rutas, umbrales, etc.)
├── README.md
│
├── interfaz/                        # Todo lo visual
│   ├── __init__.py
│   ├── ventana_principal.py         # Menú: Aprender / Traducir
│   ├── modulo_aprendizaje.py        # Parte 1: lecciones + práctica guiada
│   ├── modulo_traduccion.py         # Parte 2: cámara en vivo + predicción
│   ├── widgets/
│   │   ├── camara_widget.py
│   │   ├── teclado_visual.py        # Abecedario / vocabulario con imágenes
│   │   └── barra_progreso.py
│   └── recursos/
│       ├── iconos/
│       ├── sonidos/
│       └── imagenes_referencia/     # Fotos de cada seña para el módulo de aprendizaje
│
├── agentes/                         # Lógica "inteligente"
│   ├── __init__.py
│   ├── agente_detector_manos.py     # MediaPipe: extrae landmarks
│   ├── agente_normalizador.py       # Normaliza landmarks a vector fijo
│   ├── agente_clasificador.py       # Modelo ML (entrena + predice)
│   ├── agente_aprendizaje.py        # Lógica del módulo educativo (progreso, repetición espaciada)
│   ├── agente_voz.py                # TTS: convierte texto → audio
│   └── agente_orquestador.py        # Coordina a los demás agentes
│
├── modelos/                         # Modelos entrenados (.pkl / .h5 / .onnx)
│   ├── modelo_estatico.pkl
│   └── modelo_dinamico.h5           # (si luego haces señas en movimiento)
│
├── datasets/
│   ├── crudo/                       # Videos/imágenes originales descargados
│   ├── procesado/                   # Landmarks extraídos (.npy / .csv)
│   ├── etiquetas.csv                # label, path_landmarks
│   └── splits/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
│
├── datos_aprendizaje/               # ← lo que pediste (aprendizaje local)
│   ├── capturas_usuario/            # Muestras nuevas que graba el usuario
│   │   ├── casa/
│   │   ├── hola/
│   │   └── ...
│   ├── feedback/                    # Correcciones manuales (predije mal → etiqueta real)
│   │   └── feedback_log.csv
│   ├── historial_entrenamiento.csv  # métricas por versión del modelo
│   ├── progreso_usuario.json        # nivel, señas aprendidas, tiempo
│   └── modelos_reentrenados/        # versiones incrementales del modelo
│
├── utils/
│   ├── logger.py
│   ├── io_helpers.py                # guardar/cargar .npy, json, csv
│   └── constantes.py                # SEÑAS_VALIDAS, NUM_LANDMARKS, etc.
│
├── scripts/
│   ├── descargar_dataset.py         # ej. descarga LSA o similar
│   ├── extraer_landmarks.py         # convierte videos → landmarks
│   ├── entrenar_modelo.py           # offline
│   └── reentrenar_con_usuario.py    # usa datos_aprendizaje/
│
└── tests/
    ├── test_normalizador.py
    └── test_clasificador.py