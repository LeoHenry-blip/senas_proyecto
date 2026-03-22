"""
entrenador.py
=============
Módulo completo de entrenamiento para capturar y guardar
tus propios gestos personalizados en gestos.json.

Funcionalidades:
  - Capturar múltiples muestras de un gesto
  - Promediar muestras para mayor precisión
  - Revisar/eliminar gestos existentes
  - Mostrar estadísticas de la base de datos
  - Exportar e importar gestos entre archivos
  - Modo de prueba: detectar en tiempo real contra los gestos guardados

Uso:
    python entrenador.py
    python entrenador.py --gestos mi_base.json
    python entrenador.py --modo prueba
"""

import cv2           # OpenCV para cámara
import json          # Manejo del archivo JSON
import numpy as np   # Operaciones matemáticas
import os            # Manejo de archivos y rutas
import time          # Temporizadores y pausas
import argparse      # Argumentos de línea de comandos
import shutil        # Para copias de seguridad
from datetime import datetime  # Fecha y hora para backups
from typing import Optional, List, Tuple  # Tipos para anotaciones

# Importar módulos propios del sistema
from detector_manos import DetectorManos
from base_datos import BaseDatosGestos


# =============================================================================
# CONSTANTES DE CONFIGURACIÓN
# =============================================================================

# Número de muestras a capturar por gesto para promediar
MUESTRAS_POR_GESTO = 30

# Tiempo de cuenta regresiva antes de capturar (segundos)
TIEMPO_CUENTA_REGRESIVA = 3

# Tiempo entre captura de muestras (segundos)
INTERVALO_ENTRE_MUESTRAS = 0.05  # 20 capturas por segundo

# Resolución de la cámara de entrenamiento
CAM_ANCHO = 640
CAM_ALTO  = 480

# Colores para OpenCV (BGR)
COLOR_VERDE    = (0, 220, 80)
COLOR_ROJO     = (0, 60, 220)
COLOR_AMARILLO = (0, 210, 255)
COLOR_AZUL     = (220, 140, 0)
COLOR_BLANCO   = (255, 255, 255)
COLOR_GRIS     = (160, 160, 160)
COLOR_NEGRO    = (0, 0, 0)
COLOR_CYAN     = (255, 220, 0)


# =============================================================================
# CLASE PRINCIPAL DE ENTRENAMIENTO
# =============================================================================

class Entrenador:
    """
    Gestiona la captura, almacenamiento y revisión de gestos de entrenamiento.
    Usa la cámara para capturar landmarks de MediaPipe y los guarda en JSON.
    """

    def __init__(self, ruta_json: str = "gestos.json"):
        """
        Inicializa el entrenador con la base de datos de gestos.

        Args:
            ruta_json: Ruta al archivo JSON donde se guardan los gestos
        """
        # Ruta al archivo de gestos
        self.ruta_json = ruta_json

        # Detector de manos (MediaPipe)
        print("[Entrenador] Iniciando MediaPipe Hands...")
        self.detector = DetectorManos(
            max_manos=1,
            confianza_deteccion=0.7,
            confianza_seguimiento=0.5
        )

        # Base de datos de gestos (para comparar en modo prueba)
        self.db = BaseDatosGestos(ruta_json)

        # Captura de cámara (se abre en cada sesión)
        self.cap: Optional[cv2.VideoCapture] = None

        # Estado actual del entrenamiento
        self.gestos_data: dict = {}   # Datos cargados del JSON
        self._cargar_json()

        print(f"[Entrenador] Base de datos lista: {len(self.gestos_data.get('gestures', {}))} gestos")

    # =========================================================================
    # MANEJO DEL ARCHIVO JSON
    # =========================================================================

    def _cargar_json(self) -> None:
        """Carga el archivo JSON existente o crea uno nuevo vacío."""
        if os.path.exists(self.ruta_json):
            try:
                with open(self.ruta_json, 'r', encoding='utf-8') as f:
                    self.gestos_data = json.load(f)
                # Asegurar que existen las claves necesarias
                if 'gestures' not in self.gestos_data:
                    self.gestos_data['gestures'] = {}
                if 'metadata' not in self.gestos_data:
                    self.gestos_data['metadata'] = {}
            except Exception as e:
                print(f"[Entrenador] Error cargando JSON: {e}")
                self.gestos_data = {'version': '1.0', 'gestures': {}, 'metadata': {}}
        else:
            # Crear estructura vacía si no existe el archivo
            self.gestos_data = {
                'version': '1.0',
                'description': 'Base de datos de gestos - Lenguaje de Señas',
                'gestures': {},
                'metadata': {}
            }

    def _guardar_json(self) -> bool:
        """
        Guarda los gestos en el archivo JSON.
        Hace una copia de seguridad antes de sobreescribir.

        Returns:
            True si se guardó correctamente
        """
        try:
            # Hacer backup del archivo existente antes de sobreescribir
            if os.path.exists(self.ruta_json):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                ruta_backup = self.ruta_json.replace('.json', f'_backup_{timestamp}.json')
                shutil.copy2(self.ruta_json, ruta_backup)
                print(f"[Entrenador] Backup creado: {ruta_backup}")

            # Actualizar metadata antes de guardar
            gestures = self.gestos_data.get('gestures', {})
            letras  = [k for k, v in gestures.items() if v.get('type') == 'letter']
            palabras = [k for k, v in gestures.items() if v.get('type') == 'word']
            custom  = [k for k, v in gestures.items() if v.get('type') == 'custom']

            self.gestos_data['metadata'] = {
                'total_gestures': len(gestures),
                'letters': letras,
                'words': palabras,
                'custom': custom,
                'last_updated': datetime.now().isoformat(),
                'coordinate_format': 'normalized_x_y_z_per_landmark',
                'landmarks_per_gesture': 21
            }

            # Guardar con formato legible (indentación de 2 espacios)
            with open(self.ruta_json, 'w', encoding='utf-8') as f:
                json.dump(self.gestos_data, f, ensure_ascii=False, indent=2)

            print(f"[Entrenador] Guardado exitoso: {self.ruta_json}")
            return True

        except Exception as e:
            print(f"[Entrenador] Error guardando JSON: {e}")
            return False

    # =========================================================================
    # MANEJO DE LA CÁMARA
    # =========================================================================

    def _abrir_camara(self) -> bool:
        """
        Abre la cámara con la resolución configurada.

        Returns:
            True si la cámara se abrió correctamente
        """
        # Intentar abrir la cámara (índice 0 = principal)
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            print("[Entrenador] ERROR: No se pudo abrir la cámara")
            return False

        # Configurar resolución
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_ANCHO)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_ALTO)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # Minimizar buffer
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        print("[Entrenador] Cámara abierta correctamente")
        return True

    def _cerrar_camara(self) -> None:
        """Cierra y libera la cámara."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()

    def _leer_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Lee un frame de la cámara y lo voltea (efecto espejo).

        Returns:
            Tupla (éxito, frame) donde frame es BGR numpy array
        """
        if not self.cap or not self.cap.isOpened():
            return False, None

        exito, frame = self.cap.read()
        if not exito or frame is None:
            return False, None

        # Voltear horizontalmente para efecto espejo (más natural)
        frame = cv2.flip(frame, 1)
        return True, frame

    # =========================================================================
    # DIBUJO EN PANTALLA
    # =========================================================================

    def _dibujar_panel_info(
        self,
        frame: np.ndarray,
        titulo: str,
        lineas: List[str],
        color_titulo=COLOR_CYAN
    ) -> None:
        """
        Dibuja un panel de información con fondo semitransparente en el frame.

        Args:
            frame: Frame donde dibujar
            titulo: Título del panel
            lineas: Lista de líneas de texto a mostrar
            color_titulo: Color del título
        """
        # Fondo semitransparente en la parte superior
        overlay = frame.copy()
        alto_panel = 40 + len(lineas) * 28

        cv2.rectangle(overlay, (0, 0), (CAM_ANCHO, alto_panel), (15, 15, 20), -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

        # Título
        cv2.putText(
            frame, titulo,
            (14, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85, color_titulo, 2, cv2.LINE_AA
        )

        # Líneas de información
        for i, linea in enumerate(lineas):
            y = 58 + i * 28
            cv2.putText(
                frame, linea,
                (14, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60, COLOR_BLANCO, 1, cv2.LINE_AA
            )

    def _dibujar_barra_progreso(
        self,
        frame: np.ndarray,
        progreso: float,        # 0.0 a 1.0
        y_pos: int = 455,
        color=COLOR_VERDE
    ) -> None:
        """
        Dibuja una barra de progreso horizontal en la parte inferior.

        Args:
            frame: Frame donde dibujar
            progreso: Valor entre 0.0 y 1.0
            y_pos: Posición vertical de la barra
            color: Color de la barra
        """
        # Fondo de la barra (gris oscuro)
        cv2.rectangle(frame, (14, y_pos), (CAM_ANCHO - 14, y_pos + 18),
                      (50, 50, 55), -1)

        # Barra de progreso
        ancho_progreso = int((CAM_ANCHO - 28) * progreso)
        if ancho_progreso > 0:
            cv2.rectangle(frame, (14, y_pos),
                          (14 + ancho_progreso, y_pos + 18), color, -1)

        # Borde
        cv2.rectangle(frame, (14, y_pos), (CAM_ANCHO - 14, y_pos + 18),
                      COLOR_GRIS, 1)

    def _dibujar_cuenta_regresiva(
        self,
        frame: np.ndarray,
        segundos_restantes: float
    ) -> None:
        """
        Dibuja la cuenta regresiva grande en el centro del frame.

        Args:
            frame: Frame donde dibujar
            segundos_restantes: Tiempo restante en segundos
        """
        numero = str(int(segundos_restantes) + 1)

        # Calcular tamaño del texto para centrarlo
        (w_txt, h_txt), _ = cv2.getTextSize(
            numero, cv2.FONT_HERSHEY_SIMPLEX, 5.0, 8
        )
        x_txt = (CAM_ANCHO - w_txt) // 2
        y_txt = (CAM_ALTO + h_txt) // 2

        # Sombra del número
        cv2.putText(
            frame, numero,
            (x_txt + 4, y_txt + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            5.0, COLOR_NEGRO, 8, cv2.LINE_AA
        )
        # Número principal
        cv2.putText(
            frame, numero,
            (x_txt, y_txt),
            cv2.FONT_HERSHEY_SIMPLEX,
            5.0, COLOR_AMARILLO, 8, cv2.LINE_AA
        )

    def _dibujar_checkmark(self, frame: np.ndarray) -> None:
        """Dibuja un checkmark verde grande al completar la captura."""
        cx, cy = CAM_ANCHO // 2, CAM_ALTO // 2

        # Círculo verde de fondo
        cv2.circle(frame, (cx, cy), 70, COLOR_VERDE, -1)
        cv2.circle(frame, (cx, cy), 70, (0, 180, 60), 3)

        # Palomita (checkmark) blanca
        pts = np.array([
            [cx - 35, cy],
            [cx - 10, cy + 30],
            [cx + 40, cy - 30]
        ], dtype=np.int32)
        cv2.polylines(frame, [pts], False, COLOR_BLANCO, 8, cv2.LINE_AA)

    # =========================================================================
    # CAPTURA DE GESTOS
    # =========================================================================

    def _extraer_vector_normalizado(self, landmarks) -> Optional[np.ndarray]:
        """
        Extrae el vector de landmarks y lo normaliza.
        Igual que en base_datos.py pero independiente para el entrenador.

        Args:
            landmarks: Landmarks de MediaPipe

        Returns:
            Vector numpy normalizado o None
        """
        try:
            puntos = []
            for punto in landmarks.landmark:
                puntos.extend([punto.x, punto.y, punto.z])

            vector = np.array(puntos, dtype=np.float32)

            # Normalizar restando posición de la muñeca (punto 0)
            muñeca_x = vector[0]
            muñeca_y = vector[1]
            for i in range(21):
                vector[i * 3]     -= muñeca_x
                vector[i * 3 + 1] -= muñeca_y

            # Normalizar a longitud unitaria
            norma = np.linalg.norm(vector)
            if norma > 0:
                vector = vector / norma

            return vector

        except Exception as e:
            print(f"[Entrenador] Error extrayendo vector: {e}")
            return None

    def capturar_gesto(
        self,
        nombre: str,
        tipo: str = "letter",
        descripcion: str = "",
        num_muestras: int = MUESTRAS_POR_GESTO
    ) -> bool:
        """
        Captura un gesto nuevo con cuenta regresiva y promedio de muestras.

        Proceso:
        1. Muestra instrucciones al usuario
        2. Cuenta regresiva de 3 segundos
        3. Captura N muestras mientras la mano está visible
        4. Promedia los vectores capturados
        5. Guarda en gestos.json

        Args:
            nombre: Nombre del gesto (ej: "A", "HOLA")
            tipo: Tipo: "letter", "word" o "custom"
            descripcion: Descripción opcional del gesto
            num_muestras: Cuántas muestras capturar para promediar

        Returns:
            True si la captura fue exitosa
        """
        # Abrir la cámara
        if not self._abrir_camara():
            return False

        nombre_upper = nombre.upper().strip()
        print(f"\n[Entrenador] === Capturando gesto: '{nombre_upper}' ===")
        print(f"[Entrenador] Tipo: {tipo} | Muestras: {num_muestras}")

        # Lista donde se acumulan los vectores capturados
        muestras_capturadas: List[np.ndarray] = []

        # ---- FASE 1: Preparación (mostrar instrucciones) ----
        fase = "preparacion"         # Fase actual: preparacion / cuenta / captura / listo
        tiempo_inicio_fase = time.time()

        while True:
            # Leer frame
            exito, frame = self._leer_frame()
            if not exito:
                continue

            # Detectar manos en el frame
            frame_procesado, resultados = self.detector.detectar(frame)

            # Verificar si hay mano visible
            hay_mano = self.detector.hay_manos(resultados)

            tiempo_transcurrido = time.time() - tiempo_inicio_fase

            # ---- Lógica de fases ----
            if fase == "preparacion":
                # Mostrar instrucciones hasta que el usuario esté listo
                self._dibujar_panel_info(
                    frame_procesado,
                    f"ENTRENANDO: '{nombre_upper}'",
                    [
                        f"Tipo: {tipo}  |  Muestras a capturar: {num_muestras}",
                        "Coloca tu mano y mantén el gesto estático.",
                        "Presiona  ESPACIO  para iniciar la captura.",
                        "Presiona  ESC  para cancelar.",
                    ],
                    color_titulo=COLOR_AMARILLO
                )

                # Indicador de mano detectada
                if hay_mano:
                    cv2.putText(
                        frame_procesado, "✓ Mano detectada",
                        (CAM_ANCHO - 240, CAM_ALTO - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_VERDE, 2, cv2.LINE_AA
                    )
                else:
                    cv2.putText(
                        frame_procesado, "  Esperando mano...",
                        (CAM_ANCHO - 240, CAM_ALTO - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_ROJO, 2, cv2.LINE_AA
                    )

            elif fase == "cuenta":
                # Cuenta regresiva visual
                tiempo_restante = TIEMPO_CUENTA_REGRESIVA - tiempo_transcurrido

                if tiempo_restante <= 0:
                    # Pasar a captura
                    fase = "captura"
                    tiempo_inicio_fase = time.time()
                    muestras_capturadas.clear()
                    continue

                self._dibujar_cuenta_regresiva(frame_procesado, tiempo_restante)
                self._dibujar_panel_info(
                    frame_procesado,
                    f"PREPARATE: '{nombre_upper}'",
                    ["Mantén el gesto... ¡Ya casi!"],
                    color_titulo=COLOR_AMARILLO
                )
                self._dibujar_barra_progreso(
                    frame_procesado,
                    1.0 - (tiempo_restante / TIEMPO_CUENTA_REGRESIVA),
                    color=COLOR_AMARILLO
                )

            elif fase == "captura":
                # Capturar muestras activamente
                progreso = len(muestras_capturadas) / num_muestras

                self._dibujar_panel_info(
                    frame_procesado,
                    f"CAPTURANDO: '{nombre_upper}'",
                    [
                        f"Muestras: {len(muestras_capturadas)} / {num_muestras}",
                        "Mantén el gesto estático...",
                    ],
                    color_titulo=COLOR_VERDE
                )
                self._dibujar_barra_progreso(frame_procesado, progreso, color=COLOR_VERDE)

                # Capturar muestra si hay mano visible
                if hay_mano:
                    landmarks = self.detector.obtener_primera_mano(resultados)
                    vector = self._extraer_vector_normalizado(landmarks)

                    if vector is not None:
                        muestras_capturadas.append(vector)

                # Verificar si ya tenemos suficientes muestras
                if len(muestras_capturadas) >= num_muestras:
                    fase = "listo"
                    tiempo_inicio_fase = time.time()

                # Si no hay mano por mucho tiempo, volver a preparación
                if not hay_mano and len(muestras_capturadas) < 5:
                    # Perdimos la mano antes de empezar, reiniciar
                    muestras_capturadas.clear()
                    fase = "preparacion"
                    tiempo_inicio_fase = time.time()

            elif fase == "listo":
                # Captura completa
                self._dibujar_checkmark(frame_procesado)
                self._dibujar_panel_info(
                    frame_procesado,
                    f"¡CAPTURA COMPLETA! '{nombre_upper}'",
                    [
                        f"Se capturaron {len(muestras_capturadas)} muestras.",
                        "Presiona  ESPACIO  para guardar.",
                        "Presiona  R  para repetir.",
                        "Presiona  ESC  para cancelar.",
                    ],
                    color_titulo=COLOR_VERDE
                )

            # Mostrar frame
            cv2.imshow(f"Entrenador de Gestos - '{nombre_upper}'", frame_procesado)

            # ---- Manejo de teclas ----
            tecla = cv2.waitKey(int(INTERVALO_ENTRE_MUESTRAS * 1000)) & 0xFF

            if tecla == 27:  # ESC = cancelar
                print(f"[Entrenador] Captura cancelada por el usuario")
                self._cerrar_camara()
                return False

            elif tecla == 32:  # ESPACIO
                if fase == "preparacion":
                    # Iniciar cuenta regresiva
                    if hay_mano:
                        fase = "cuenta"
                        tiempo_inicio_fase = time.time()
                    else:
                        print("[Entrenador] ¡Pon tu mano frente a la cámara primero!")

                elif fase == "listo":
                    # Guardar el gesto
                    break

            elif tecla == ord('r') or tecla == ord('R'):
                # R = repetir captura
                if fase == "listo":
                    muestras_capturadas.clear()
                    fase = "preparacion"
                    tiempo_inicio_fase = time.time()

        # Cerrar cámara
        self._cerrar_camara()

        # ---- Calcular vector promedio de todas las muestras ----
        if not muestras_capturadas:
            print("[Entrenador] ERROR: No se capturaron muestras válidas")
            return False

        # Apilar todos los vectores en una matriz y promediar por columna
        matriz = np.stack(muestras_capturadas, axis=0)   # (N, 63)
        vector_promedio = np.mean(matriz, axis=0)          # (63,)

        # Re-normalizar el vector promediado
        norma = np.linalg.norm(vector_promedio)
        if norma > 0:
            vector_promedio = vector_promedio / norma

        # ---- Guardar en la estructura JSON ----
        self.gestos_data['gestures'][nombre_upper] = {
            "name": nombre_upper,
            "type": tipo,
            "description": descripcion if descripcion else f"Gesto capturado: {nombre_upper}",
            "landmarks": vector_promedio.tolist(),
            "muestras_usadas": len(muestras_capturadas),
            "fecha_captura": datetime.now().isoformat()
        }

        # Guardar en archivo
        exito = self._guardar_json()

        if exito:
            # Recargar la base de datos en memoria para que el modo prueba funcione
            self.db = BaseDatosGestos(self.ruta_json)
            print(f"[Entrenador] ✅ Gesto '{nombre_upper}' guardado exitosamente")
            print(f"[Entrenador]    Muestras promediadas: {len(muestras_capturadas)}")

        return exito

    # =========================================================================
    # MODO PRUEBA EN TIEMPO REAL
    # =========================================================================

    def modo_prueba(self) -> None:
        """
        Modo de prueba: detecta gestos en tiempo real y muestra qué reconoce.
        Útil para verificar la calidad del entrenamiento.
        """
        if not self._abrir_camara():
            return

        # Recargar la base de datos con los gestos más recientes
        self.db = BaseDatosGestos(self.ruta_json)

        if not self.db.gestos:
            print("[Entrenador] No hay gestos entrenados. Entrena primero.")
            self._cerrar_camara()
            return

        print(f"\n[Entrenador] MODO PRUEBA - {len(self.db.gestos)} gestos cargados")
        print("[Entrenador] Presiona ESC para salir")

        # Historial de las últimas detecciones para suavizado visual
        historial_detecciones = []

        while True:
            exito, frame = self._leer_frame()
            if not exito:
                continue

            # Detectar manos
            frame_procesado, resultados = self.detector.detectar(frame)

            gesto_detectado = "---"
            confianza = 0.0

            # Si hay mano, reconocer el gesto
            if self.detector.hay_manos(resultados):
                landmarks = self.detector.obtener_primera_mano(resultados)
                vector = self._extraer_vector_normalizado(landmarks)

                if vector is not None:
                    # Buscar el gesto más cercano
                    nombre, conf = self.db.buscar_gesto(vector)
                    if nombre:
                        gesto_detectado = nombre
                        confianza = conf
                        historial_detecciones.append(nombre)
                        if len(historial_detecciones) > 15:
                            historial_detecciones.pop(0)

            # Suavizado: mostrar el gesto más votado en el historial
            gesto_final = gesto_detectado
            if historial_detecciones:
                from collections import Counter
                conteo = Counter(historial_detecciones)
                gesto_final = conteo.most_common(1)[0][0]

            # Color de la confianza
            if confianza > 0.7:
                color_conf = COLOR_VERDE
            elif confianza > 0.4:
                color_conf = COLOR_AMARILLO
            else:
                color_conf = COLOR_ROJO

            # Panel de información
            self._dibujar_panel_info(
                frame_procesado,
                "MODO PRUEBA  (ESC para salir)",
                [
                    f"Gesto detectado:  {gesto_final}",
                    f"Confianza: {int(confianza * 100)}%   |   Gestos en DB: {len(self.db.gestos)}",
                    "Realiza un gesto frente a la cámara para probarlo.",
                ],
                color_titulo=COLOR_CYAN
            )

            # Mostrar el gesto detectado en grande en el centro
            if gesto_final != "---":
                # Fondo semitransparente para el texto grande
                overlay = frame_procesado.copy()
                cv2.rectangle(overlay, (CAM_ANCHO // 2 - 100, CAM_ALTO - 130),
                              (CAM_ANCHO // 2 + 100, CAM_ALTO - 50), (15, 15, 20), -1)
                cv2.addWeighted(overlay, 0.7, frame_procesado, 0.3, 0, frame_procesado)

                cv2.putText(
                    frame_procesado, gesto_final,
                    (CAM_ANCHO // 2 - 80, CAM_ALTO - 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2.2, color_conf, 4, cv2.LINE_AA
                )

            # Barra de confianza
            self._dibujar_barra_progreso(
                frame_procesado, confianza,
                y_pos=CAM_ALTO - 30,
                color=color_conf
            )

            cv2.imshow("Entrenador - Modo Prueba", frame_procesado)

            # ESC para salir
            if cv2.waitKey(1) & 0xFF == 27:
                break

        self._cerrar_camara()
        print("[Entrenador] Modo prueba finalizado")

    # =========================================================================
    # GESTIÓN DE LA BASE DE DATOS
    # =========================================================================

    def listar_gestos(self) -> None:
        """Muestra todos los gestos guardados en la base de datos."""
        gestures = self.gestos_data.get('gestures', {})

        if not gestures:
            print("\n[Entrenador] La base de datos está vacía.")
            return

        print("\n" + "=" * 60)
        print(f"  BASE DE DATOS: {len(gestures)} gestos")
        print("=" * 60)

        # Separar por tipo
        letras  = {k: v for k, v in gestures.items() if v.get('type') == 'letter'}
        palabras = {k: v for k, v in gestures.items() if v.get('type') == 'word'}
        custom  = {k: v for k, v in gestures.items() if v.get('type') == 'custom'}

        # Mostrar letras
        if letras:
            print(f"\n📝 LETRAS ({len(letras)}):")
            for nombre, datos in sorted(letras.items()):
                muestras = datos.get('muestras_usadas', '?')
                fecha = datos.get('fecha_captura', 'desconocida')[:10]
                print(f"   {nombre:6} | muestras: {str(muestras):4} | {fecha}")

        # Mostrar palabras
        if palabras:
            print(f"\n💬 PALABRAS ({len(palabras)}):")
            for nombre, datos in sorted(palabras.items()):
                muestras = datos.get('muestras_usadas', '?')
                desc = datos.get('description', '')[:40]
                print(f"   {nombre:10} | {desc}")

        # Mostrar custom
        if custom:
            print(f"\n⚙️  PERSONALIZADOS ({len(custom)}):")
            for nombre, datos in sorted(custom.items()):
                desc = datos.get('description', '')[:40]
                print(f"   {nombre:10} | {desc}")

        print("=" * 60)

    def eliminar_gesto(self, nombre: str) -> bool:
        """
        Elimina un gesto de la base de datos.

        Args:
            nombre: Nombre del gesto a eliminar

        Returns:
            True si se eliminó correctamente
        """
        nombre_upper = nombre.upper().strip()
        gestures = self.gestos_data.get('gestures', {})

        if nombre_upper not in gestures:
            print(f"[Entrenador] Gesto '{nombre_upper}' no encontrado")
            return False

        # Confirmar eliminación
        confirmar = input(f"¿Eliminar '{nombre_upper}'? (s/n): ").strip().lower()
        if confirmar != 's':
            print("[Entrenador] Eliminación cancelada")
            return False

        del gestures[nombre_upper]
        exito = self._guardar_json()

        if exito:
            # Recargar base de datos
            self.db = BaseDatosGestos(self.ruta_json)
            print(f"[Entrenador] ✅ Gesto '{nombre_upper}' eliminado")

        return exito

    def exportar_gestos(self, ruta_destino: str) -> bool:
        """
        Exporta la base de datos actual a un archivo diferente.

        Args:
            ruta_destino: Ruta donde exportar

        Returns:
            True si se exportó correctamente
        """
        try:
            with open(ruta_destino, 'w', encoding='utf-8') as f:
                json.dump(self.gestos_data, f, ensure_ascii=False, indent=2)
            print(f"[Entrenador] ✅ Exportado a: {ruta_destino}")
            return True
        except Exception as e:
            print(f"[Entrenador] Error exportando: {e}")
            return False

    def importar_gestos(self, ruta_origen: str, sobreescribir: bool = False) -> int:
        """
        Importa gestos desde otro archivo JSON, fusionándolos con los actuales.

        Args:
            ruta_origen: Ruta del archivo a importar
            sobreescribir: Si True, sobreescribe gestos con el mismo nombre

        Returns:
            Cantidad de gestos importados
        """
        if not os.path.exists(ruta_origen):
            print(f"[Entrenador] Archivo no encontrado: {ruta_origen}")
            return 0

        try:
            with open(ruta_origen, 'r', encoding='utf-8') as f:
                datos_import = json.load(f)

            gestures_import = datos_import.get('gestures', {})
            gestures_actual  = self.gestos_data.get('gestures', {})

            importados = 0
            omitidos = 0

            for nombre, datos in gestures_import.items():
                if nombre in gestures_actual and not sobreescribir:
                    omitidos += 1
                    continue
                gestures_actual[nombre] = datos
                importados += 1

            # Guardar la fusión
            self._guardar_json()
            self.db = BaseDatosGestos(self.ruta_json)

            print(f"[Entrenador] ✅ Importados: {importados} gestos")
            if omitidos:
                print(f"[Entrenador]    Omitidos (ya existen): {omitidos}")

            return importados

        except Exception as e:
            print(f"[Entrenador] Error importando: {e}")
            return 0

    def liberar(self) -> None:
        """Libera todos los recursos."""
        self._cerrar_camara()
        self.detector.liberar()


# =============================================================================
# MENÚ INTERACTIVO EN CONSOLA
# =============================================================================

def mostrar_menu() -> None:
    """Muestra el menú principal del entrenador."""
    print("\n" + "=" * 55)
    print("   🤟  ENTRENADOR DE LENGUAJE DE SEÑAS")
    print("=" * 55)
    print("  1. Capturar nueva LETRA     (A, B, C...)")
    print("  2. Capturar nueva PALABRA   (HOLA, GRACIAS...)")
    print("  3. Capturar gesto CUSTOM    (cualquier nombre)")
    print("  4. Ver gestos guardados")
    print("  5. Eliminar un gesto")
    print("  6. Modo PRUEBA en tiempo real")
    print("  7. Exportar base de datos")
    print("  8. Importar gestos desde archivo")
    print("  9. Entrenar abecedario completo (A-Z)")
    print("  0. Salir")
    print("=" * 55)


def entrenar_abecedario(entrenador: Entrenador) -> None:
    """
    Guía al usuario para entrenar todo el abecedario (A-Z) de forma secuencial.

    Args:
        entrenador: Instancia del Entrenador
    """
    letras_pendientes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    # Filtrar letras que ya están entrenadas
    gestures_existentes = entrenador.gestos_data.get('gestures', {})
    letras_ya_entrenadas = [l for l in letras_pendientes if l in gestures_existentes]

    if letras_ya_entrenadas:
        print(f"\n[Entrenador] Letras ya entrenadas: {', '.join(letras_ya_entrenadas)}")
        continuar_todas = input("¿Re-entrenar las ya existentes también? (s/n): ").strip().lower()
        if continuar_todas != 's':
            letras_pendientes = [l for l in letras_pendientes if l not in letras_ya_entrenadas]

    if not letras_pendientes:
        print("[Entrenador] ¡Todo el abecedario ya está entrenado!")
        return

    print(f"\n[Entrenador] Se entrenarán {len(letras_pendientes)} letras: {', '.join(letras_pendientes)}")
    input("Presiona ENTER para comenzar...")

    entrenadas = 0
    for i, letra in enumerate(letras_pendientes):
        print(f"\n[Entrenador] Letra {i+1}/{len(letras_pendientes)}: '{letra}'")
        print(f"[Entrenador] Quedan: {', '.join(letras_pendientes[i+1:])}")

        exito = entrenador.capturar_gesto(
            nombre=letra,
            tipo="letter",
            descripcion=f"Letra {letra} del abecedario",
            num_muestras=MUESTRAS_POR_GESTO
        )

        if exito:
            entrenadas += 1
            print(f"[Entrenador] ✅ '{letra}' guardada ({entrenadas}/{len(letras_pendientes)})")
        else:
            # Preguntar si continuar o salir
            decision = input(f"La captura de '{letra}' falló. ¿Continuar con la siguiente? (s/n): ")
            if decision.strip().lower() != 's':
                break

    print(f"\n[Entrenador] Sesión completada: {entrenadas}/{len(letras_pendientes)} letras entrenadas")


def main():
    """Función principal: menú interactivo del entrenador."""

    # Parsear argumentos de línea de comandos
    parser = argparse.ArgumentParser(description="Entrenador de Gestos para Lenguaje de Señas")
    parser.add_argument('--gestos', type=str, default='gestos.json',
                        help='Ruta al archivo JSON de gestos (default: gestos.json)')
    parser.add_argument('--modo', type=str, choices=['prueba', 'menu'],
                        default='menu', help='Modo de inicio: prueba o menu')
    args = parser.parse_args()

    print("\n🤟 Iniciando Entrenador de Lenguaje de Señas...")

    # Crear instancia del entrenador
    entrenador = Entrenador(ruta_json=args.gestos)

    # Si se pidió modo prueba directamente, lanzarlo
    if args.modo == 'prueba':
        entrenador.modo_prueba()
        entrenador.liberar()
        return

    # ---- Menú interactivo ----
    while True:
        mostrar_menu()

        try:
            opcion = input("\n  Elige una opción (0-9): ").strip()
        except KeyboardInterrupt:
            print("\n\n[Entrenador] Ctrl+C detectado. Saliendo...")
            break

        if opcion == '0':
            break

        elif opcion == '1':
            # Capturar letra
            letra = input("  Letra a capturar (A-Z): ").strip().upper()
            if len(letra) == 1 and letra.isalpha():
                desc = input(f"  Descripción opcional para '{letra}' (Enter para omitir): ").strip()
                entrenador.capturar_gesto(letra, tipo="letter", descripcion=desc)
            else:
                print("  [!] Ingresa solo una letra del abecedario")

        elif opcion == '2':
            # Capturar palabra
            palabra = input("  Palabra a capturar (ej: HOLA): ").strip().upper()
            if palabra:
                desc = input(f"  Descripción opcional para '{palabra}': ").strip()
                entrenador.capturar_gesto(palabra, tipo="word", descripcion=desc)

        elif opcion == '3':
            # Capturar gesto custom
            nombre = input("  Nombre del gesto custom: ").strip().upper()
            if nombre:
                desc = input(f"  Descripción de '{nombre}': ").strip()
                entrenador.capturar_gesto(nombre, tipo="custom", descripcion=desc)

        elif opcion == '4':
            # Ver gestos guardados
            entrenador.listar_gestos()
            input("\n  Presiona ENTER para continuar...")

        elif opcion == '5':
            # Eliminar gesto
            entrenador.listar_gestos()
            nombre = input("\n  Nombre del gesto a eliminar: ").strip().upper()
            if nombre:
                entrenador.eliminar_gesto(nombre)

        elif opcion == '6':
            # Modo prueba
            entrenador.modo_prueba()

        elif opcion == '7':
            # Exportar
            destino = input("  Ruta de destino (ej: mis_gestos.json): ").strip()
            if destino:
                entrenador.exportar_gestos(destino)

        elif opcion == '8':
            # Importar
            origen = input("  Ruta del archivo a importar: ").strip()
            sobreescribir = input("  ¿Sobreescribir gestos duplicados? (s/n): ").strip().lower() == 's'
            if origen:
                entrenador.importar_gestos(origen, sobreescribir=sobreescribir)

        elif opcion == '9':
            # Entrenar abecedario completo
            entrenar_abecedario(entrenador)

        else:
            print("  [!] Opción no válida. Elige entre 0 y 9.")

    # Liberar recursos al salir
    entrenador.liberar()
    print("\n[Entrenador] ¡Hasta luego! 🤟\n")


# Punto de entrada
if __name__ == "__main__":
    main()
