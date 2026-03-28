"""
reconocedor.py  v3.1
====================
Motor de reconocimiento con soporte hibrido:

  LETRAS  → suavizado por buffer de 10 frames + euclidiana
  PALABRAS → ventana deslizante de frames + DTW

La ventana deslizante captura el movimiento en tiempo real:
  - Acumula los ultimos N frames mientras hay mano visible
  - Cuando detecta inicio de movimiento, empieza a grabar
  - Cuando el movimiento termina (pausa), evalua con DTW
  - Si DTW reconoce una palabra, la agrega a la frase

Convivencia letras/palabras:
  El sistema detecta letras y palabras en paralelo.
  Si DTW da una palabra con confianza > umbral, la palabra gana.
  Si no, se sigue acumulando letras normalmente.

Cambios v3.1:
  - Fix: _dtw_inicio_t se resetea correctamente al limpiar la ventana
"""

import time
import numpy as np
from collections import deque
from typing import Optional, Tuple, List

from base_datos import BaseDatosGestos, NormalizadorMano


class Reconocedor:
    """
    Reconoce gestos (letras y palabras) y forma frases en tiempo real.

    Flujo para LETRAS:
      frame → normalizar → buscar euclidiana → buffer → confirmar → agregar

    Flujo para PALABRAS:
      frame → normalizar → ventana deslizante → DTW → confirmar → agregar
    """

    def __init__(self, ruta_gestos: str = "gestos.json"):
        # Base de datos (letras + palabras)
        self.db = BaseDatosGestos(ruta_gestos)

        # ── Configuracion de suavizado para letras ──────────────────────
        self.buffer_tamano = 10
        self.buffer_gestos = deque(maxlen=self.buffer_tamano)

        # ── Configuracion de tiempos ────────────────────────────────────
        self.tiempo_confirmacion  = 1.2   # Segundos sosteniendo un gesto estatico
        self.tiempo_pausa_letra   = 1.5   # Pausa entre letras
        self.tiempo_pausa_palabra = 3.0   # Pausa para cerrar palabra

        # ── Estado de texto ─────────────────────────────────────────────
        self.letra_actual:    str   = ""
        self.palabra_actual:  str   = ""
        self.frase_completa:  str   = ""
        self.confianza_actual: float = 0.0

        # ── Control de tiempos ──────────────────────────────────────────
        self.ultimo_gesto_tiempo  = time.time()
        self.ultima_letra_tiempo  = time.time()
        self.gesto_inicio_tiempo  = time.time()
        self.ultimo_gesto_nombre: Optional[str] = None
        self.gesto_confirmado:    bool = False

        # ── Pausa detectada (para corrector IA) ─────────────────────────
        self.pausa_detectada:     bool = False
        self.frase_para_corregir: str  = ""

        # ── VENTANA DESLIZANTE PARA DTW ──────────────────────────────────
        self.frames_dtw_max   = 30    # Maximo frames a acumular
        self.frames_dtw_min   = 8     # Minimo frames para evaluar DTW
        self.ventana_dtw: deque = deque(maxlen=self.frames_dtw_max)

        # Estado de la ventana DTW
        self._dtw_activo       = False   # True = hay movimiento en curso
        self._dtw_inicio_t     = 0.0     # Cuando empezo el movimiento
        self._dtw_ultima_eval  = 0.0     # Ultima vez que se evaluo DTW
        self._dtw_intervalo    = 0.5     # Evaluar DTW cada N segundos

        # Ultimo resultado DTW (para no repetir)
        self._ultima_palabra_dtw: Optional[str] = None
        self._dtw_palabra_tiempo  = 0.0

        print("[Reconocedor] Sistema inicializado (v3.1 con DTW)")

    # =========================================================================
    # PROCESAMIENTO DE LANDMARKS (llamado en cada frame)
    # =========================================================================

    def procesar_landmarks(self, landmarks) -> Tuple[Optional[str], float]:
        """
        Procesa landmarks de MediaPipe y retorna el gesto de letra reconocido.
        Tambien actualiza la ventana DTW para palabras.

        Args:
            landmarks: Landmarks de MediaPipe

        Returns:
            (nombre_letra, confianza) — solo letras, las palabras se manejan
            internamente via ventana DTW
        """
        vector = self.db.extraer_vector_mano(landmarks)
        if vector is None:
            return None, 0.0

        # Agregar frame a la ventana DTW
        # FIX v3.1: _dtw_inicio_t se asigna solo cuando la ventana estaba vacía
        # (no cuando ya está activa), evitando que quede congelado indefinidamente
        if not self._dtw_activo:
            self._dtw_inicio_t = time.time()
        self._dtw_activo = True
        self.ventana_dtw.append(vector)

        # Buscar la letra mas parecida
        nombre, confianza = self.db.buscar_gesto(vector)

        # Agregar al buffer de suavizado de letras
        self.buffer_gestos.append((nombre, confianza))

        if len(self.buffer_gestos) < 3:
            return None, 0.0

        # Suavizado por votacion
        conteo:         dict = {}
        suma_confianza: dict = {}
        for g, c in self.buffer_gestos:
            if g is not None:
                conteo[g]         = conteo.get(g, 0) + 1
                suma_confianza[g] = suma_confianza.get(g, 0.0) + c

        if not conteo:
            return None, 0.0

        gesto_suavizado    = max(conteo, key=conteo.get)
        confianza_promedio = suma_confianza[gesto_suavizado] / conteo[gesto_suavizado]

        votos_minimos = max(3, self.buffer_tamano * 0.4)
        if conteo[gesto_suavizado] < votos_minimos:
            return None, 0.0

        return gesto_suavizado, confianza_promedio

    # =========================================================================
    # ACTUALIZAR CON MANO (llamado en cada frame con mano visible)
    # =========================================================================

    def actualizar_con_mano(self, nombre_gesto: Optional[str],
                             confianza: float) -> bool:
        """
        Actualiza el estado cuando hay mano visible.
        Evalua DTW periodicamente si hay suficientes frames acumulados.

        Returns:
            True si se agrego algo al texto
        """
        ahora = time.time()
        self.ultimo_gesto_tiempo = ahora

        # ── Evaluar DTW periodicamente ───────────────────────────────────
        if (len(self.ventana_dtw) >= self.frames_dtw_min and
                ahora - self._dtw_ultima_eval >= self._dtw_intervalo):
            self._dtw_ultima_eval = ahora
            self._evaluar_dtw()

        # ── Flujo normal de letras ────────────────────────────────────────
        if nombre_gesto is None:
            self.letra_actual    = ""
            self.confianza_actual = 0.0
            return False

        self.letra_actual    = nombre_gesto
        self.confianza_actual = confianza

        if nombre_gesto != self.ultimo_gesto_nombre:
            self.ultimo_gesto_nombre = nombre_gesto
            self.gesto_inicio_tiempo = ahora
            self.gesto_confirmado    = False
            return False

        tiempo_sosteniendo  = ahora - self.gesto_inicio_tiempo
        tiempo_desde_ultima = ahora - self.ultima_letra_tiempo

        if (tiempo_sosteniendo  >= self.tiempo_confirmacion and
                tiempo_desde_ultima >= self.tiempo_pausa_letra and
                not self.gesto_confirmado):

            self.gesto_confirmado = True
            self._agregar_al_texto(nombre_gesto)
            self.ultima_letra_tiempo = ahora
            return True

        return False

    # =========================================================================
    # EVALUAR DTW (interno)
    # =========================================================================

    def _evaluar_dtw(self) -> None:
        """
        Evalua la ventana DTW actual contra todas las palabras entrenadas.
        Si encuentra una coincidencia con suficiente confianza, agrega
        la palabra a la frase y limpia la ventana.
        """
        if not self.db.palabras:
            return

        secuencia = np.stack(list(self.ventana_dtw), axis=0)
        nombre, confianza = self.db.buscar_palabra_dtw(secuencia)

        if nombre is None or confianza < 0.55:
            return

        ahora = time.time()

        # Evitar repetir la misma palabra en menos de 2 segundos
        if (nombre == self._ultima_palabra_dtw and
                ahora - self._dtw_palabra_tiempo < 2.0):
            return

        print(f"[Reconocedor] Palabra DTW reconocida: '{nombre}' "
              f"(confianza={confianza:.2f})")

        self._agregar_palabra_directa(nombre)

        self._ultima_palabra_dtw = nombre
        self._dtw_palabra_tiempo = ahora
        self.ventana_dtw.clear()
        self._dtw_activo   = False
        self._dtw_inicio_t = 0.0   # FIX v3.1: resetear al limpiar la ventana

    # =========================================================================
    # ACTUALIZAR SIN MANO
    # =========================================================================

    def actualizar_sin_mano(self) -> None:
        """
        Llama cuando no hay mano visible.
        Maneja pausas para separar letras y palabras.
        Evalua DTW final cuando termina el movimiento.
        """
        ahora = time.time()
        tiempo_sin_mano = ahora - self.ultimo_gesto_tiempo

        self.buffer_gestos.clear()
        self.letra_actual    = ""
        self.confianza_actual = 0.0

        # Evaluar DTW al terminar el movimiento (pausa breve = fin del gesto)
        if (self._dtw_activo and
                len(self.ventana_dtw) >= self.frames_dtw_min and
                tiempo_sin_mano >= 0.3):
            self._evaluar_dtw()
            self._dtw_activo = False

        # Limpiar ventana DTW si pasa mucho tiempo sin mano
        if tiempo_sin_mano >= 1.5:
            self.ventana_dtw.clear()
            self._dtw_activo   = False
            self._dtw_inicio_t = 0.0   # FIX v3.1: resetear correctamente

        # Pausa para finalizar palabra
        if tiempo_sin_mano >= self.tiempo_pausa_palabra:
            if self.palabra_actual:
                if self.frase_completa:
                    self.frase_completa += " " + self.palabra_actual
                else:
                    self.frase_completa = self.palabra_actual

                print(f"[Reconocedor] Palabra: '{self.palabra_actual}' → "
                      f"Frase: '{self.frase_completa}'")
                self.palabra_actual = ""

                if self.frase_completa:
                    self.pausa_detectada     = True
                    self.frase_para_corregir = self.frase_completa

    # =========================================================================
    # AGREGAR TEXTO
    # =========================================================================

    def _agregar_al_texto(self, nombre: str) -> None:
        """Agrega una letra al texto en formacion."""
        tipo = self.db.gestos.get(nombre, {}).get("tipo", "letter")

        if tipo == "word":
            self._agregar_palabra_directa(nombre)
        else:
            self.palabra_actual += nombre
            print(f"[Reconocedor] Letra: '{nombre}' → "
                  f"Palabra: '{self.palabra_actual}'")

    def _agregar_palabra_directa(self, nombre: str) -> None:
        """
        Agrega una palabra completa a la frase.
        Primero cierra cualquier letra en formacion.
        """
        if self.palabra_actual:
            if self.frase_completa:
                self.frase_completa += " " + self.palabra_actual
            else:
                self.frase_completa = self.palabra_actual
            self.palabra_actual = ""

        if self.frase_completa:
            self.frase_completa += " " + nombre
        else:
            self.frase_completa = nombre

        print(f"[Reconocedor] Palabra directa: '{nombre}' → "
              f"Frase: '{self.frase_completa}'")

    # =========================================================================
    # ESTADO Y CONTROL
    # =========================================================================

    def obtener_estado(self) -> dict:
        """Estado actual para mostrar en la UI."""
        return {
            "letra_actual":          self.letra_actual,
            "palabra_actual":        self.palabra_actual,
            "frase_completa":        self.frase_completa,
            "confianza":             self.confianza_actual,
            "pausa_detectada":       self.pausa_detectada,
            "frase_para_corregir":   self.frase_para_corregir,
            "frames_dtw_acumulados": len(self.ventana_dtw),
            "dtw_activo":            self._dtw_activo,
        }

    def limpiar_todo(self) -> None:
        """Reinicia todo el estado."""
        self.letra_actual        = ""
        self.palabra_actual      = ""
        self.frase_completa      = ""
        self.confianza_actual    = 0.0
        self.ultimo_gesto_nombre = None
        self.gesto_confirmado    = False
        self.pausa_detectada     = False
        self.frase_para_corregir = ""
        self.buffer_gestos.clear()
        self.ventana_dtw.clear()
        self._dtw_activo          = False
        self._dtw_inicio_t        = 0.0
        self._ultima_palabra_dtw  = None
        print("[Reconocedor] Estado limpiado")

    def consumir_pausa(self) -> Optional[str]:
        """Consume la pausa detectada y retorna la frase para corregir."""
        if self.pausa_detectada:
            frase = self.frase_para_corregir
            self.pausa_detectada     = False
            self.frase_para_corregir = ""
            return frase
        return None

    def forzar_fin_palabra(self) -> None:
        """Fuerza el cierre de la palabra actual (boton manual)."""
        if self.palabra_actual:
            if self.frase_completa:
                self.frase_completa += " " + self.palabra_actual
            else:
                self.frase_completa = self.palabra_actual
            self.palabra_actual      = ""
            self.pausa_detectada     = True
            self.frase_para_corregir = self.frase_completa