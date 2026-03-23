"""
entrenador.py  v3.0
====================
Herramienta de entrenamiento con soporte DTW para palabras con movimiento.

Modos de captura:
  - LETRA (letter/custom): captura estatica, 30 muestras promediadas
  - PALABRA (word): captura dinamica, graba secuencia de 10 frames
                    durante el movimiento, repite 3 veces para robustez

Uso:
    python entrenador.py
    python entrenador.py --modo prueba
"""

import cv2
import json
import numpy as np
import os
import time
import argparse
import shutil
from datetime import datetime
from typing import Optional, List, Tuple
from collections import deque

from detector_manos import DetectorManos
from base_datos import BaseDatosGestos, NormalizadorMano, dtw_distancia_rapida


# ─── Constantes ───────────────────────────────────────────────────────────────
MUESTRAS_LETRA      = 30     # Frames para promediar en letras
REPETICIONES_PALABRA = 3     # Cuantas veces grabar cada palabra
FRAMES_SECUENCIA    = 10     # Frames de la secuencia DTW
CUENTA_REGRESIVA    = 3      # Segundos antes de empezar a capturar
INTERVALO_CAPTURA   = 0.05   # Segundos entre frames capturados

CAM_ANCHO, CAM_ALTO = 640, 480

# Colores BGR
VERDE    = (0,   210, 80)
ROJO     = (0,   60,  220)
AMARILLO = (0,   210, 255)
AZUL     = (220, 140, 0)
BLANCO   = (255, 255, 255)
GRIS     = (160, 160, 160)
NEGRO    = (0,   0,   0)
CYAN     = (255, 200, 0)


class Entrenador:
    """
    Herramienta de entrenamiento con soporte para DTW.
    Detecta automaticamente si el gesto es estatico (letra) o
    dinamico (palabra) segun el tipo elegido.
    """

    def __init__(self, ruta_json: str = "gestos.json"):
        self.ruta_json  = ruta_json
        self.detector   = DetectorManos(max_manos=1, confianza_deteccion=0.7)
        self.db         = BaseDatosGestos(ruta_json)
        self.cap: Optional[cv2.VideoCapture] = None
        self._cargar_json()

    # ──────────────────────────────────────────────────────────────────────────
    # JSON
    # ──────────────────────────────────────────────────────────────────────────

    def _cargar_json(self) -> None:
        if os.path.exists(self.ruta_json):
            try:
                with open(self.ruta_json, 'r', encoding='utf-8') as f:
                    self.gestos_data = json.load(f)
                if 'gestures' not in self.gestos_data:
                    self.gestos_data['gestures'] = {}
            except Exception:
                self.gestos_data = {'version': '3.0', 'gestures': {}}
        else:
            self.gestos_data = {'version': '3.0', 'gestures': {}}

    def _guardar_json(self) -> bool:
        try:
            if os.path.exists(self.ruta_json):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                shutil.copy2(self.ruta_json,
                             self.ruta_json.replace('.json', f'_backup_{ts}.json'))
            with open(self.ruta_json, 'w', encoding='utf-8') as f:
                json.dump(self.gestos_data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"[Entrenador] Error guardando: {e}")
            return False

    # ──────────────────────────────────────────────────────────────────────────
    # CAMARA
    # ──────────────────────────────────────────────────────────────────────────

    def _abrir_camara(self) -> bool:
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_ANCHO)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_ALTO)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return True

    def _cerrar_camara(self) -> None:
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()

    def _leer_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self.cap or not self.cap.isOpened():
            return False, None
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return False, None
        return True, cv2.flip(frame, 1)

    # ──────────────────────────────────────────────────────────────────────────
    # DIBUJO
    # ──────────────────────────────────────────────────────────────────────────

    def _panel(self, frame, titulo, lineas, color=CYAN):
        overlay = frame.copy()
        h = 42 + len(lineas) * 28
        cv2.rectangle(overlay, (0, 0), (CAM_ANCHO, h), (15, 15, 20), -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
        cv2.putText(frame, titulo, (14, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)
        for i, l in enumerate(lineas):
            cv2.putText(frame, l, (14, 58 + i*28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, BLANCO, 1, cv2.LINE_AA)

    def _barra(self, frame, progreso, y=455, color=VERDE):
        cv2.rectangle(frame, (14, y), (CAM_ANCHO-14, y+18), (50, 50, 55), -1)
        w = int((CAM_ANCHO - 28) * max(0, min(1, progreso)))
        if w > 0:
            cv2.rectangle(frame, (14, y), (14+w, y+18), color, -1)
        cv2.rectangle(frame, (14, y), (CAM_ANCHO-14, y+18), GRIS, 1)

    def _cuenta(self, frame, segs_restantes):
        num = str(int(segs_restantes) + 1)
        (w, h), _ = cv2.getTextSize(num, cv2.FONT_HERSHEY_SIMPLEX, 5.0, 8)
        x, y = (CAM_ANCHO-w)//2, (CAM_ALTO+h)//2
        cv2.putText(frame, num, (x+4, y+4),
                    cv2.FONT_HERSHEY_SIMPLEX, 5.0, NEGRO, 8, cv2.LINE_AA)
        cv2.putText(frame, num, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 5.0, AMARILLO, 8, cv2.LINE_AA)

    # ──────────────────────────────────────────────────────────────────────────
    # CAPTURA DE LETRA (estatica)
    # ──────────────────────────────────────────────────────────────────────────

    def capturar_letra(self, nombre: str, tipo: str = "letter",
                        descripcion: str = "") -> bool:
        """
        Captura un gesto estatico (letra) promediando 30 muestras.
        Compatible con el sistema anterior.
        """
        if not self._abrir_camara():
            return False

        nombre_u = nombre.upper().strip()
        muestras: List[np.ndarray] = []
        fase      = "preparacion"
        t_inicio  = time.time()

        while True:
            ok, frame = self._leer_frame()
            if not ok:
                continue

            _, resultados = self.detector.detectar(frame)
            hay_mano      = self.detector.hay_manos(resultados)
            ahora         = time.time()

            if fase == "preparacion":
                self._panel(frame, f"LETRA: '{nombre_u}'",
                    ["Coloca tu mano y manten el gesto estatico.",
                     "Presiona  ESPACIO  para iniciar.",
                     "Presiona  ESC  para cancelar."], AMARILLO)
                msg = "Mano detectada" if hay_mano else "Sin mano..."
                col = VERDE if hay_mano else ROJO
                cv2.putText(frame, msg, (CAM_ANCHO-230, CAM_ALTO-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)

            elif fase == "cuenta":
                restante = CUENTA_REGRESIVA - (ahora - t_inicio)
                if restante <= 0:
                    fase = "captura"
                    t_inicio = ahora
                    muestras.clear()
                    continue
                self._cuenta(frame, restante)
                self._barra(frame, 1-(restante/CUENTA_REGRESIVA), color=AMARILLO)

            elif fase == "captura":
                progreso = len(muestras) / MUESTRAS_LETRA
                self._panel(frame, f"CAPTURANDO: '{nombre_u}'",
                    [f"Muestras: {len(muestras)} / {MUESTRAS_LETRA}",
                     "Manten el gesto estatico..."], VERDE)
                self._barra(frame, progreso)

                if hay_mano:
                    lm = self.detector.obtener_primera_mano(resultados)
                    v  = NormalizadorMano.normalizar(lm)
                    if v is not None:
                        muestras.append(v)

                if len(muestras) >= MUESTRAS_LETRA:
                    fase = "listo"

                if not hay_mano and len(muestras) < 5:
                    muestras.clear()
                    fase    = "preparacion"
                    t_inicio = ahora

            elif fase == "listo":
                self._panel(frame, f"LISTO: '{nombre_u}'",
                    [f"Capturadas {len(muestras)} muestras.",
                     "ESPACIO = guardar   |   R = repetir   |   ESC = cancelar"], VERDE)

            cv2.imshow(f"Entrenador — '{nombre_u}'", frame)
            tecla = cv2.waitKey(int(INTERVALO_CAPTURA * 1000)) & 0xFF

            if tecla == 27:
                self._cerrar_camara()
                return False
            elif tecla == 32:
                if fase == "preparacion" and hay_mano:
                    fase = "cuenta"
                    t_inicio = ahora
                elif fase == "listo":
                    break
            elif tecla in (ord('r'), ord('R')) and fase == "listo":
                muestras.clear()
                fase = "preparacion"

        self._cerrar_camara()
        if not muestras:
            return False

        # Promediar y guardar
        promedio = np.mean(np.stack(muestras), axis=0)
        norma    = np.linalg.norm(promedio)
        if norma > 1e-6:
            promedio = promedio / norma

        self.gestos_data['gestures'][nombre_u] = {
            "name":          nombre_u,
            "type":          tipo,
            "description":   descripcion or f"Letra {nombre_u}",
            "landmarks":     promedio.tolist(),
            "muestras_usadas": len(muestras),
            "fecha_captura": datetime.now().isoformat(),
        }
        exito = self._guardar_json()
        if exito:
            self.db = BaseDatosGestos(self.ruta_json)
            print(f"[Entrenador] Letra '{nombre_u}' guardada OK")
        return exito

    # ──────────────────────────────────────────────────────────────────────────
    # CAPTURA DE PALABRA (dinamica con DTW)
    # ──────────────────────────────────────────────────────────────────────────

    def capturar_palabra(self, nombre: str,
                          descripcion: str = "") -> bool:
        """
        Captura una palabra con movimiento usando DTW.

        Proceso por cada repeticion (3 veces total):
          1. Cuenta regresiva de 3 segundos
          2. Graba mientras hay mano visible (maximo 2 segundos)
          3. Guarda la secuencia de frames

        Las 3 secuencias se guardan como muestras de entrenamiento.
        Esto hace el DTW mas robusto a variaciones de velocidad.
        """
        if not self._abrir_camara():
            return False

        nombre_u     = nombre.upper().strip()
        secuencias:  List[np.ndarray] = []   # Lista de secuencias capturadas
        rep_actual   = 0
        fase         = "intro"
        t_inicio     = time.time()
        frames_rep:  List[np.ndarray] = []   # Frames de la repeticion actual
        grabando     = False

        print(f"\n[Entrenador] Iniciando captura DTW de: '{nombre_u}'")
        print(f"[Entrenador] Repeticiones necesarias: {REPETICIONES_PALABRA}")

        while True:
            ok, frame = self._leer_frame()
            if not ok:
                continue

            _, resultados = self.detector.detectar(frame)
            hay_mano      = self.detector.hay_manos(resultados)
            ahora         = time.time()

            # ── Intro: explicar al usuario ──────────────────────────────
            if fase == "intro":
                self._panel(frame, f"PALABRA CON MOVIMIENTO: '{nombre_u}'",
                    ["Este modo captura el MOVIMIENTO completo de la senya.",
                     f"Necesitas repetirla {REPETICIONES_PALABRA} veces.",
                     "Haz el movimiento COMPLETO de la senya.",
                     "Presiona  ESPACIO  para empezar.",
                     "Presiona  ESC  para cancelar."], CYAN)
                # Indicador de repeticiones
                for i in range(REPETICIONES_PALABRA):
                    color = VERDE if i < rep_actual else GRIS
                    cv2.circle(frame, (30 + i*30, CAM_ALTO-25), 10, color, -1)

            # ── Cuenta regresiva antes de cada rep ─────────────────────
            elif fase == "cuenta":
                restante = CUENTA_REGRESIVA - (ahora - t_inicio)
                if restante <= 0:
                    fase    = "grabar"
                    t_inicio = ahora
                    frames_rep.clear()
                    grabando = False
                    continue

                self._panel(frame,
                    f"Rep {rep_actual+1}/{REPETICIONES_PALABRA}: PREPARATE",
                    [f"Palabra: '{nombre_u}'",
                     "Cuando suene, haz el movimiento completo de la senya."],
                    AMARILLO)
                self._cuenta(frame, restante)
                self._barra(frame, 1-(restante/CUENTA_REGRESIVA), color=AMARILLO)

            # ── Grabando la secuencia ───────────────────────────────────
            elif fase == "grabar":
                tiempo_grabando = ahora - t_inicio
                max_tiempo      = 2.5   # Maximo 2.5 segundos de grabacion

                if hay_mano:
                    grabando = True
                    lm  = self.detector.obtener_primera_mano(resultados)
                    v   = NormalizadorMano.normalizar(lm)
                    if v is not None:
                        frames_rep.append(v)

                # Calcular cuantos frames queremos (resamplear a 10)
                progreso = min(len(frames_rep) / FRAMES_SECUENCIA, 1.0)

                self._panel(frame,
                    f"GRABANDO rep {rep_actual+1}/{REPETICIONES_PALABRA}",
                    [f"Frames: {len(frames_rep)}  |  Haz la senya ahora!",
                     f"La grabacion para automaticamente al terminar."],
                    ROJO)
                self._barra(frame, progreso, color=ROJO)

                # Dibujar indicador de grabacion
                if grabando and int(ahora * 2) % 2 == 0:
                    cv2.circle(frame, (CAM_ANCHO-30, 30), 12, ROJO, -1)
                    cv2.putText(frame, "REC", (CAM_ANCHO-70, 38),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, ROJO, 2)

                # Terminar grabacion: cuando deja de haber mano O tiempo limite
                terminar = (
                    (grabando and not hay_mano and len(frames_rep) >= 5) or
                    tiempo_grabando >= max_tiempo
                )

                if terminar:
                    if len(frames_rep) >= 5:
                        # Resamplear a exactamente FRAMES_SECUENCIA frames
                        seq = self._resamplear(frames_rep, FRAMES_SECUENCIA)
                        secuencias.append(seq)
                        rep_actual += 1
                        print(f"[Entrenador] Rep {rep_actual}: "
                              f"{len(frames_rep)} frames → {FRAMES_SECUENCIA} resampled")
                    else:
                        print(f"[Entrenador] Rep descartada: muy pocos frames "
                              f"({len(frames_rep)})")

                    frames_rep.clear()
                    grabando = False

                    if rep_actual >= REPETICIONES_PALABRA:
                        fase = "listo"
                    else:
                        fase     = "entre_rep"
                        t_inicio = ahora

            # ── Pausa entre repeticiones ─────────────────────────────────
            elif fase == "entre_rep":
                restante = 1.5 - (ahora - t_inicio)
                if restante <= 0:
                    fase     = "cuenta"
                    t_inicio = ahora
                    continue

                self._panel(frame,
                    f"Rep {rep_actual}/{REPETICIONES_PALABRA} lista!",
                    [f"Siguiente en {restante:.1f} segundos...",
                     "Descansa un momento."], VERDE)
                # Mostrar progreso de repeticiones
                for i in range(REPETICIONES_PALABRA):
                    color = VERDE if i < rep_actual else GRIS
                    cv2.circle(frame, (30 + i*40, CAM_ALTO-25), 12, color, -1)
                    cv2.putText(frame, str(i+1), (24+i*40, CAM_ALTO-20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLANCO, 1)

            # ── Listo: guardar ───────────────────────────────────────────
            elif fase == "listo":
                self._panel(frame,
                    f"CAPTURA COMPLETA: '{nombre_u}'",
                    [f"Se grabaron {len(secuencias)} secuencias de movimiento.",
                     "ESPACIO = guardar   |   R = repetir todo   |   ESC = cancelar"],
                    VERDE)
                for i in range(REPETICIONES_PALABRA):
                    color = VERDE if i < len(secuencias) else GRIS
                    cv2.circle(frame, (30 + i*40, CAM_ALTO-25), 12, color, -1)

            cv2.imshow(f"Entrenador DTW — '{nombre_u}'", frame)
            tecla = cv2.waitKey(int(INTERVALO_CAPTURA * 1000)) & 0xFF

            if tecla == 27:
                self._cerrar_camara()
                return False
            elif tecla == 32:
                if fase == "intro":
                    fase     = "cuenta"
                    t_inicio = ahora
                elif fase == "listo":
                    break
            elif tecla in (ord('r'), ord('R')) and fase == "listo":
                # Repetir todo desde cero
                secuencias.clear()
                rep_actual = 0
                fase       = "intro"

        self._cerrar_camara()

        if not secuencias:
            print("[Entrenador] No se capturaron secuencias validas")
            return False

        # Guardar en JSON
        seqs_lista = [seq.tolist() for seq in secuencias]

        # Si ya existe, agregar las nuevas secuencias (mas robustez)
        existente = self.gestos_data['gestures'].get(nombre_u, {})
        seqs_prev = existente.get("secuencia", [])
        seqs_total = seqs_prev + seqs_lista

        self.gestos_data['gestures'][nombre_u] = {
            "name":        nombre_u,
            "type":        "word",
            "description": descripcion or f"Palabra con movimiento: {nombre_u}",
            "secuencia":   seqs_total,
            # Primer frame del primer entrenamiento como landmark legacy
            "landmarks":   seqs_lista[0][0] if seqs_lista else [],
            "repeticiones": len(seqs_total),
            "frames_por_secuencia": FRAMES_SECUENCIA,
            "fecha_captura": datetime.now().isoformat(),
        }

        exito = self._guardar_json()
        if exito:
            self.db = BaseDatosGestos(self.ruta_json)
            print(f"[Entrenador] Palabra '{nombre_u}' guardada con DTW "
                  f"({len(seqs_total)} secuencias)")
        return exito

    # ──────────────────────────────────────────────────────────────────────────
    # RESAMPLEAR SECUENCIA
    # ──────────────────────────────────────────────────────────────────────────

    def _resamplear(self, frames: List[np.ndarray], n_target: int) -> np.ndarray:
        """
        Resamplea una secuencia de frames a exactamente n_target frames.
        Usa interpolacion lineal para no perder informacion del movimiento.

        Args:
            frames:   Lista de vectores capturados (longitud variable)
            n_target: Numero de frames destino (10)

        Returns:
            Array (n_target, 63) resampled
        """
        n_orig = len(frames)
        if n_orig == n_target:
            return np.stack(frames, axis=0)

        # Indices de origen (0 a n_orig-1) mapeados a destino (0 a n_target-1)
        indices_orig   = np.linspace(0, n_orig - 1, n_target)
        stack_orig     = np.stack(frames, axis=0)   # (n_orig, 63)
        resultado      = np.zeros((n_target, 63), dtype=np.float32)

        for i, idx in enumerate(indices_orig):
            i_bajo  = int(idx)
            i_alto  = min(i_bajo + 1, n_orig - 1)
            fraccion = idx - i_bajo

            # Interpolacion lineal entre dos frames consecutivos
            resultado[i] = (stack_orig[i_bajo] * (1 - fraccion) +
                             stack_orig[i_alto] * fraccion)

        return resultado

    # ──────────────────────────────────────────────────────────────────────────
    # MODO PRUEBA
    # ──────────────────────────────────────────────────────────────────────────

    def modo_prueba(self) -> None:
        """
        Prueba el reconocimiento en tiempo real.
        Muestra letras (euclidiana) y palabras (DTW) con su confianza.
        """
        if not self._abrir_camara():
            return

        self.db = BaseDatosGestos(self.ruta_json)
        if not self.db.gestos and not self.db.palabras:
            print("[Entrenador] No hay gestos. Entrena primero.")
            self._cerrar_camara()
            return

        print(f"[Entrenador] MODO PRUEBA: {len(self.db.gestos)} letras, "
              f"{len(self.db.palabras)} palabras")
        print("[Entrenador] ESC para salir")

        # Ventana deslizante para DTW
        ventana_dtw: deque = deque(maxlen=30)
        hist_letras: deque = deque(maxlen=15)

        while True:
            ok, frame = self._leer_frame()
            if not ok:
                continue

            _, resultados = self.detector.detectar(frame)
            gesto_txt     = "---"
            conf          = 0.0
            tipo_txt      = ""
            nombre_palabra = None
            conf_palabra   = 0.0

            if self.detector.hay_manos(resultados):
                lm = self.detector.obtener_primera_mano(resultados)
                v  = NormalizadorMano.normalizar(lm)

                if v is not None:
                    # Letra
                    nombre_l, conf_l = self.db.buscar_gesto(v)
                    # Acumular en ventana DTW
                    ventana_dtw.append(v)

                    # Palabra (si hay suficientes frames)
                    if len(ventana_dtw) >= 8:
                        seq            = np.stack(list(ventana_dtw), axis=0)
                        nombre_palabra, conf_palabra = self.db.buscar_palabra_dtw(seq)

                    # Elegir que mostrar
                    if conf_palabra > 0.55 and nombre_palabra:
                        gesto_txt = nombre_palabra
                        conf      = conf_palabra
                        tipo_txt  = "PALABRA (DTW)"
                        hist_letras.append(nombre_palabra)
                        ventana_dtw.clear()
                    elif nombre_l:
                        gesto_txt = nombre_l
                        conf      = conf_l
                        tipo_txt  = "letra"
                        hist_letras.append(nombre_l)
            else:
                ventana_dtw.clear()

            # UI
            col_conf = VERDE if conf > 0.7 else (AMARILLO if conf > 0.4 else ROJO)
            self._panel(frame, "MODO PRUEBA  (ESC para salir)",
                [f"Gesto: {gesto_txt}  [{tipo_txt}]",
                 f"Confianza: {int(conf*100)}%  |  "
                 f"Letras:{len(self.db.gestos)}  Palabras:{len(self.db.palabras)}",
                 f"Frames DTW acumulados: {len(ventana_dtw)}"],
                CYAN)

            # Gesto grande en centro
            if gesto_txt != "---":
                cv2.putText(frame, gesto_txt,
                            (CAM_ANCHO//2-80, CAM_ALTO-65),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.2, col_conf, 4, cv2.LINE_AA)

            self._barra(frame, conf, CAM_ALTO-30, col_conf)

            # Historial
            hist_txt = "  ".join(list(hist_letras)[-10:])
            cv2.putText(frame, hist_txt, (14, CAM_ALTO-38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, GRIS, 1, cv2.LINE_AA)

            cv2.imshow("Entrenador — Modo Prueba", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        self._cerrar_camara()

    # ──────────────────────────────────────────────────────────────────────────
    # GESTIÓN
    # ──────────────────────────────────────────────────────────────────────────

    def listar_gestos(self) -> None:
        gestures = self.gestos_data.get('gestures', {})
        letras   = {k: v for k, v in gestures.items() if v.get('type') != 'word'}
        palabras = {k: v for k, v in gestures.items() if v.get('type') == 'word'}

        print(f"\n{'='*55}")
        print(f"  BASE DE DATOS: {len(gestures)} gestos")
        print(f"{'='*55}")

        if letras:
            print(f"\n📝 LETRAS ({len(letras)}):")
            for n, d in sorted(letras.items()):
                print(f"   {n:6} | muestras: {d.get('muestras_usadas','?')}")

        if palabras:
            print(f"\n💬 PALABRAS CON MOVIMIENTO — DTW ({len(palabras)}):")
            for n, d in sorted(palabras.items()):
                n_seqs = len(d.get('secuencia', []))
                fps    = d.get('frames_por_secuencia', FRAMES_SECUENCIA)
                print(f"   {n:12} | {n_seqs} secuencias x {fps} frames")

        print(f"{'='*55}")

    def eliminar_gesto(self, nombre: str) -> bool:
        n = nombre.upper().strip()
        if n not in self.gestos_data.get('gestures', {}):
            print(f"[Entrenador] Gesto '{n}' no encontrado")
            return False
        conf = input(f"Eliminar '{n}'? (s/n): ").strip().lower()
        if conf != 's':
            return False
        del self.gestos_data['gestures'][n]
        exito = self._guardar_json()
        if exito:
            self.db = BaseDatosGestos(self.ruta_json)
            print(f"[Entrenador] '{n}' eliminado")
        return exito

    def liberar(self) -> None:
        self._cerrar_camara()
        self.detector.liberar()


# =============================================================================
# MENÚ
# =============================================================================

def mostrar_menu():
    print(f"\n{'='*55}")
    print(f"   ENTRENADOR DE SENYAS V2  (con DTW para palabras)")
    print(f"{'='*55}")
    print("  1. Capturar LETRA       (estatica: A, B, C...)")
    print("  2. Capturar PALABRA     (con movimiento: HOLA, GRACIAS...)")
    print("  3. Ver gestos guardados")
    print("  4. Eliminar un gesto")
    print("  5. Modo PRUEBA en tiempo real")
    print("  0. Salir")
    print(f"{'='*55}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gestos', default='gestos.json')
    parser.add_argument('--modo', choices=['prueba', 'menu'], default='menu')
    args = parser.parse_args()

    print("\n Iniciando Entrenador V3 (con DTW para palabras con movimiento)...")
    e = Entrenador(ruta_json=args.gestos)

    if args.modo == 'prueba':
        e.modo_prueba()
        e.liberar()
        return

    while True:
        mostrar_menu()
        try:
            op = input("\n  Elige opcion (0-5): ").strip()
        except KeyboardInterrupt:
            break

        if op == '0':
            break
        elif op == '1':
            letra = input("  Letra (A-Z): ").strip().upper()
            if len(letra) == 1 and letra.isalpha():
                desc = input(f"  Descripcion (Enter=omitir): ").strip()
                e.capturar_letra(letra, tipo="letter", descripcion=desc)
        elif op == '2':
            palabra = input("  Nombre de la palabra (ej: HOLA): ").strip().upper()
            if palabra:
                desc = input(f"  Descripcion (Enter=omitir): ").strip()
                e.capturar_palabra(palabra, descripcion=desc)
        elif op == '3':
            e.listar_gestos()
            input("\n  Enter para continuar...")
        elif op == '4':
            e.listar_gestos()
            n = input("\n  Nombre a eliminar: ").strip().upper()
            if n:
                e.eliminar_gesto(n)
        elif op == '5':
            e.modo_prueba()

    e.liberar()
    print("\n Hasta luego!\n")


if __name__ == "__main__":
    main()
