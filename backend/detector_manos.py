"""
detector_manos.py
=================
Módulo para detectar y rastrear manos usando MediaPipe.
Extrae los 21 puntos de cada mano y los dibuja en pantalla.
Optimizado para bajo consumo de recursos.
"""

import cv2           # OpenCV para procesamiento de imagen
import mediapipe as mp  # MediaPipe para detección de manos
import numpy as np   # Para operaciones matemáticas
from typing import Optional, List, Tuple  # Tipos para anotaciones


class DetectorManos:
    """
    Clase que encapsula la detección de manos con MediaPipe.
    Optimizada para laptops de recursos medios.
    """

    def __init__(
        self,
        max_manos: int = 2,
        confianza_deteccion: float = 0.7,
        confianza_seguimiento: float = 0.5
    ):
        """
        Inicializa el detector de manos de MediaPipe.
        
        Args:
            max_manos: Número máximo de manos a detectar (1 o 2)
            confianza_deteccion: Umbral mínimo para detectar una mano
            confianza_seguimiento: Umbral mínimo para seguir una mano
        """
        # Inicializar módulo de dibujo de MediaPipe
        self.mp_dibujo = mp.solutions.drawing_utils
        
        # Inicializar módulo de estilos de dibujo
        self.mp_estilos = mp.solutions.drawing_styles
        
        # Inicializar módulo de manos de MediaPipe
        self.mp_manos = mp.solutions.hands
        
        # Crear el detector con los parámetros dados
        # static_image_mode=False: optimizado para video en tiempo real
        # max_num_hands: cuántas manos detectar como máximo
        # min_detection_confidence: qué tan seguro debe estar para detectar
        # min_tracking_confidence: qué tan seguro para seguir la mano
        self.detector = self.mp_manos.Hands(
            static_image_mode=False,
            max_num_hands=max_manos,
            min_detection_confidence=confianza_deteccion,
            min_tracking_confidence=confianza_seguimiento,
            model_complexity=0  # 0 = ligero, 1 = completo (usamos ligero)
        )
        
        # Guardar los últimos resultados de detección
        # Esto permite usar el último frame si el actual falla
        self.ultimo_resultado = None
        
        # Contador de frames procesados (para estadísticas)
        self.frames_procesados = 0
        
        print("[Detector] MediaPipe Hands inicializado")

    def detectar(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[object]]:
        """
        Procesa un frame y detecta las manos.
        
        Args:
            frame: Imagen BGR de OpenCV
            
        Returns:
            Tupla (frame_con_landmarks, resultados_mediapipe)
        """
        # Incrementar contador de frames
        self.frames_procesados += 1
        
        # Convertir de BGR (OpenCV) a RGB (MediaPipe requiere RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Optimización: marcar la imagen como no escribible para mejorar velocidad
        frame_rgb.flags.writeable = False
        
        # Procesar el frame con MediaPipe
        resultados = self.detector.process(frame_rgb)
        
        # Restaurar escritura para poder dibujar encima
        frame_rgb.flags.writeable = True
        
        # Convertir de vuelta a BGR para OpenCV
        frame_procesado = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        # Si se detectaron manos, dibujar los landmarks
        if resultados.multi_hand_landmarks:
            # Iterar sobre cada mano detectada
            for landmarks_mano in resultados.multi_hand_landmarks:
                # Dibujar los 21 puntos y las conexiones entre ellos
                self.mp_dibujo.draw_landmarks(
                    frame_procesado,           # Frame donde dibujar
                    landmarks_mano,            # Puntos de la mano
                    self.mp_manos.HAND_CONNECTIONS,  # Líneas de conexión
                    # Estilo de los puntos (color blanco brillante)
                    self.mp_dibujo.DrawingSpec(
                        color=(255, 255, 255),  # Blanco
                        thickness=2,            # Grosor del punto
                        circle_radius=4         # Radio del círculo
                    ),
                    # Estilo de las conexiones (líneas cyan)
                    self.mp_dibujo.DrawingSpec(
                        color=(0, 255, 255),    # Cyan
                        thickness=2,            # Grosor de la línea
                        circle_radius=2
                    )
                )
            
            # Guardar el último resultado válido
            self.ultimo_resultado = resultados
        
        return frame_procesado, resultados

    def obtener_manos(self, resultados) -> List[object]:
        """
        Extrae la lista de landmarks de todas las manos detectadas.
        
        Args:
            resultados: Resultados de MediaPipe
            
        Returns:
            Lista de landmarks por mano (puede estar vacía)
        """
        # Si hay manos detectadas, retornarlas como lista
        if resultados and resultados.multi_hand_landmarks:
            return resultados.multi_hand_landmarks
        
        # Si no hay manos, retornar lista vacía
        return []

    def obtener_primera_mano(self, resultados) -> Optional[object]:
        """
        Retorna los landmarks de la primera mano detectada.
        Útil cuando solo se necesita una mano.
        
        Args:
            resultados: Resultados de MediaPipe
            
        Returns:
            Landmarks de la primera mano o None
        """
        manos = self.obtener_manos(resultados)
        if manos:
            return manos[0]  # Primera mano en la lista
        return None

    def hay_manos(self, resultados) -> bool:
        """
        Verifica si hay al menos una mano detectada.
        
        Args:
            resultados: Resultados de MediaPipe
            
        Returns:
            True si hay al menos una mano
        """
        return bool(resultados and resultados.multi_hand_landmarks)

    def obtener_punto_clave(
        self, 
        landmarks, 
        indice: int, 
        ancho: int, 
        alto: int
    ) -> Tuple[int, int]:
        """
        Obtiene las coordenadas en píxeles de un punto específico.
        
        Args:
            landmarks: Landmarks de MediaPipe de una mano
            indice: Índice del punto (0-20)
            ancho: Ancho del frame en píxeles
            alto: Alto del frame en píxeles
            
        Returns:
            Tupla (x, y) en píxeles
        """
        # Obtener el punto en coordenadas normalizadas (0-1)
        punto = landmarks.landmark[indice]
        
        # Convertir a píxeles multiplicando por dimensiones del frame
        x = int(punto.x * ancho)
        y = int(punto.y * alto)
        
        return x, y

    def liberar(self) -> None:
        """
        Libera los recursos de MediaPipe.
        Siempre llamar al cerrar la aplicación.
        """
        self.detector.close()
        print(f"[Detector] Recursos liberados. Frames procesados: {self.frames_procesados}")
