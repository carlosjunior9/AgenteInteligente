"""Agente de voz.

Convierte la palabra en audio. Orden de intento:
  1. pyttsx3 (offline, usa una voz en español si el sistema la tiene).
  2. gTTS (necesita internet) + reproductor del sistema.
Las frases se reproducen en un hilo y en orden, para que no se encimen.
"""
import os
import platform
import shutil
import subprocess
import tempfile
import threading

from utils.logger import obtener_logger

log = obtener_logger(__name__)


class AgenteVoz:
    def __init__(self, cfg_voz: dict):
        self.cfg = cfg_voz
        self._lock = threading.Lock()

    @property
    def activa(self) -> bool:
        return bool(self.cfg.get("activa", True))

    def hablar(self, texto: str, bloquear: bool = False) -> bool:
        if not texto:
            return False
        hilo = threading.Thread(target=self._hablar_sincrono, args=(texto,), daemon=True)
        hilo.start()
        if bloquear:
            hilo.join()
        return True

    def _hablar_sincrono(self, texto: str) -> None:
        with self._lock:
            motor = self.cfg.get("motor", "auto")
            if motor in ("auto", "pyttsx3") and self._con_pyttsx3(texto):
                return
            if motor in ("auto", "gtts") and self._con_gtts(texto):
                return
            log.warning("No hay motor de voz disponible. Instala pyttsx3 o gTTS.")

    def _con_pyttsx3(self, texto: str) -> bool:
        try:
            import pyttsx3
        except ImportError:
            return False
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", self.cfg.get("velocidad", 160))
            idioma = self.cfg.get("idioma", "es")
            for voz in engine.getProperty("voices"):
                etiqueta = f"{voz.id} {voz.name} {voz.languages}".lower()
                if idioma in etiqueta:
                    engine.setProperty("voice", voz.id)
                    break
            engine.say(texto)
            engine.runAndWait()
            engine.stop()
            return True
        except Exception as e:
            log.warning("pyttsx3 falló: %s", e)
            return False

    def _con_gtts(self, texto: str) -> bool:
        try:
            from gtts import gTTS
        except ImportError:
            return False
        try:
            archivo = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
            gTTS(text=texto, lang=self.cfg.get("idioma", "es")).save(archivo)
            _reproducir(archivo)
            return True
        except Exception as e:
            log.warning("gTTS falló (¿sin internet?): %s", e)
            return False


def _reproducir(archivo: str) -> None:
    sistema = platform.system()
    if sistema == "Windows":
        os.startfile(archivo)  # no bloquea; el archivo temporal se deja en el sistema
    elif sistema == "Darwin":
        subprocess.run(["afplay", archivo], check=False)
    else:
        for cmd in (["mpg123", "-q"], ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]):
            if shutil.which(cmd[0]):
                subprocess.run(cmd + [archivo], check=False)
                return
        raise RuntimeError("Instala mpg123 o ffmpeg para reproducir audio con gTTS.")
