"""
reconocedor.py
==============
Módulo de reconocimiento de gestos y formación de palabras/frases.
Detecta letras, forma palabras con pausas, construye frases completas.
Incluye suavizado por promedio de frames para mayor estabilidad.
"""

import time           # Para medir pausas entre letras
import numpy as np    # Para operaciones matemáticas
from collections import deque  # Cola eficiente para historial
from typing import Optional, Tuple, List  # Tipos para anotaciones

from base_datos import BaseDatosGestos  # Importar manejador de base de datos


class Reconocedor:
    """
    Reconoce gestos y construye palabras/frases a partir de letras detectadas.
    
    Flujo:
    1. Recibe landmarks de una mano
    2. Extrae y normaliza el vector
    3. Compara con la base de datos
    4. Agrega letra a la palabra actual
    5. Detecta pausa -> finaliza palabra
    6. Acumula palabras -> forma frase
    """

    def __init__(self, ruta_gestos: str = "gestos.json"):
        """
        Inicializa el reconocedor con todos sus parámetros.
        
        Args:
            ruta_gestos: Ruta al archivo JSON de gestos
        """
        # Cargar la base de datos de gestos
        self.db = BaseDatosGestos(ruta_gestos)
        
        # --- Configuración de suavizado ---
        # Buffer circular para guardar los últimos N gestos detectados
        # Esto evita que cambios rápidos/erróneos afecten el resultado
        self.buffer_tamano = 10           # Cantidad de frames a promediar
        self.buffer_gestos = deque(maxlen=self.buffer_tamano)  # Buffer FIFO
        
        # --- Configuración de tiempos ---
        # Tiempo mínimo que debe mantenerse un gesto para ser aceptado (seg)
        self.tiempo_confirmacion = 1.2
        
        # Tiempo de pausa para separar letras dentro de una palabra (seg)
        self.tiempo_pausa_letra = 1.5
        
        # Tiempo de pausa para separar palabras en la frase (seg)
        self.tiempo_pausa_palabra = 3.0
        
        # --- Estado del texto ---
        self.letra_actual: str = ""         # Letra que se está mostrando ahora
        self.palabra_actual: str = ""       # Palabra en formación
        self.frase_completa: str = ""       # Frase acumulada
        self.confianza_actual: float = 0.0  # Confianza del gesto actual
        
        # --- Control de tiempos ---
        self.ultimo_gesto_tiempo = time.time()     # Cuándo se vio el último gesto
        self.ultima_letra_tiempo = time.time()     # Cuándo se agregó la última letra
        self.gesto_inicio_tiempo = time.time()     # Cuándo empezó el gesto actual
        
        # --- Último gesto confirmado ---
        self.ultimo_gesto_nombre: Optional[str] = None  # Nombre del último gesto
        self.gesto_confirmado: bool = False              # Si ya fue añadido al texto
        
        # --- Estado de pausa detectada ---
        # True cuando el usuario hizo una pausa (para activar corrección IA)
        self.pausa_detectada: bool = False
        self.frase_para_corregir: str = ""  # Frase lista para ser corregida
        
        print("[Reconocedor] Sistema inicializado")

    def procesar_landmarks(self, landmarks) -> Tuple[Optional[str], float]:
        """
        Procesa los landmarks de una mano y retorna el gesto reconocido.
        Aplica suavizado por promedio de múltiples frames.
        
        Args:
            landmarks: Landmarks de MediaPipe de una mano
            
        Returns:
            Tupla (nombre_gesto, confianza) o (None, 0.0)
        """
        # Extraer el vector normalizado de los landmarks
        vector = self.db.extraer_vector_mano(landmarks)
        if vector is None:
            # Si falla la extracción, retornar sin resultado
            return None, 0.0
        
        # Buscar el gesto más cercano en la base de datos
        nombre, confianza = self.db.buscar_gesto(vector)
        
        # Agregar resultado al buffer de suavizado
        self.buffer_gestos.append((nombre, confianza))
        
        # Si no hay suficientes frames en el buffer, no procesar aún
        if len(self.buffer_gestos) < 3:
            return None, 0.0
        
        # --- Suavizado: encontrar el gesto más votado en el buffer ---
        # Contar cuántas veces aparece cada gesto en el buffer
        conteo: dict = {}
        suma_confianza: dict = {}
        
        for gesto_buf, conf_buf in self.buffer_gestos:
            if gesto_buf is not None:
                # Incrementar contador del gesto
                conteo[gesto_buf] = conteo.get(gesto_buf, 0) + 1
                # Acumular confianza para promediar
                suma_confianza[gesto_buf] = suma_confianza.get(gesto_buf, 0.0) + conf_buf
        
        # Si no hay votos válidos, retornar sin resultado
        if not conteo:
            return None, 0.0
        
        # Encontrar el gesto con más votos (mayoría)
        gesto_suavizado = max(conteo, key=conteo.get)
        
        # Calcular confianza promedio de ese gesto
        confianza_promedio = suma_confianza[gesto_suavizado] / conteo[gesto_suavizado]
        
        # Solo retornar si tiene suficientes votos (al menos 40% del buffer)
        votos_minimos = max(3, self.buffer_tamano * 0.4)
        if conteo[gesto_suavizado] < votos_minimos:
            return None, 0.0
        
        return gesto_suavizado, confianza_promedio

    def actualizar_sin_mano(self) -> None:
        """
        Llama cuando no hay mano detectada en el frame.
        Maneja las pausas para separar letras y palabras.
        """
        tiempo_ahora = time.time()
        
        # Calcular cuánto tiempo lleva sin detectarse la mano
        tiempo_sin_mano = tiempo_ahora - self.ultimo_gesto_tiempo
        
        # Limpiar el buffer y la letra actual
        self.buffer_gestos.clear()
        self.letra_actual = ""
        self.confianza_actual = 0.0
        
        # --- Pausa para separar palabras ---
        if tiempo_sin_mano >= self.tiempo_pausa_palabra:
            # Si hay una palabra en formación, agregarla a la frase
            if self.palabra_actual:
                # Agregar espacio entre palabras
                if self.frase_completa:
                    self.frase_completa += " " + self.palabra_actual
                else:
                    self.frase_completa = self.palabra_actual
                
                print(f"[Reconocedor] Palabra finalizada: '{self.palabra_actual}'")
                print(f"[Reconocedor] Frase: '{self.frase_completa}'")
                
                # Limpiar palabra para empezar una nueva
                self.palabra_actual = ""
                
                # Marcar que hay una frase lista para corregir
                if self.frase_completa:
                    self.pausa_detectada = True
                    self.frase_para_corregir = self.frase_completa

    def actualizar_con_mano(self, nombre_gesto: Optional[str], confianza: float) -> bool:
        """
        Actualiza el estado cuando se detecta un gesto.
        
        Args:
            nombre_gesto: Nombre del gesto detectado (puede ser None)
            confianza: Nivel de confianza (0-1)
            
        Returns:
            True si se agregó una nueva letra/palabra
        """
        tiempo_ahora = time.time()
        
        # Actualizar tiempo del último gesto válido
        self.ultimo_gesto_tiempo = tiempo_ahora
        
        # Si no se reconoció ningún gesto, salir
        if nombre_gesto is None:
            self.letra_actual = ""
            self.confianza_actual = 0.0
            return False
        
        # Actualizar la letra actual que se muestra en pantalla
        self.letra_actual = nombre_gesto
        self.confianza_actual = confianza
        
        # --- Lógica de confirmación de gesto ---
        # Solo añadir letra si el gesto es diferente al último añadido
        # O si ha pasado suficiente tiempo desde la última letra
        
        if nombre_gesto != self.ultimo_gesto_nombre:
            # Es un gesto diferente, reiniciar temporizador
            self.ultimo_gesto_nombre = nombre_gesto
            self.gesto_inicio_tiempo = tiempo_ahora
            self.gesto_confirmado = False
            return False
        
        # Es el mismo gesto, verificar si ha durado suficiente tiempo
        tiempo_sosteniendo = tiempo_ahora - self.gesto_inicio_tiempo
        
        # Verificar que pasó el tiempo mínimo entre letras
        tiempo_desde_ultima = tiempo_ahora - self.ultima_letra_tiempo
        
        # Solo confirmar si:
        # 1. Se sostuvo el gesto por el tiempo mínimo
        # 2. Pasó el tiempo mínimo desde la última letra
        # 3. No fue ya confirmado este ciclo
        if (tiempo_sosteniendo >= self.tiempo_confirmacion and
                tiempo_desde_ultima >= self.tiempo_pausa_letra and
                not self.gesto_confirmado):
            
            # Marcar como confirmado para no añadir dos veces
            self.gesto_confirmado = True
            
            # Agregar la letra/palabra al texto
            self._agregar_al_texto(nombre_gesto)
            
            # Actualizar tiempo de última letra
            self.ultima_letra_tiempo = tiempo_ahora
            
            return True  # Indica que se añadió algo nuevo
        
        return False

    def _agregar_al_texto(self, nombre: str) -> None:
        """
        Agrega un gesto reconocido al texto (letra o palabra completa).
        
        Args:
            nombre: Nombre del gesto a agregar
        """
        # Verificar el tipo de gesto en la base de datos
        if nombre in self.db.gestos:
            tipo = self.db.gestos[nombre]["tipo"]
        else:
            tipo = "letter"  # Asumir letra por defecto
        
        if tipo == "word":
            # Es una palabra completa: agregarla directamente a la frase
            if self.palabra_actual:
                # Primero finalizar la palabra en formación
                if self.frase_completa:
                    self.frase_completa += " " + self.palabra_actual
                else:
                    self.frase_completa = self.palabra_actual
                self.palabra_actual = ""
            
            # Agregar la palabra completa a la frase
            if self.frase_completa:
                self.frase_completa += " " + nombre
            else:
                self.frase_completa = nombre
            
            print(f"[Reconocedor] Palabra directa: '{nombre}'")
        
        else:
            # Es una letra: agregarla a la palabra en formación
            self.palabra_actual += nombre
            print(f"[Reconocedor] Letra añadida: '{nombre}' -> Palabra: '{self.palabra_actual}'")

    def obtener_estado(self) -> dict:
        """
        Retorna el estado actual del reconocedor para mostrar en la UI.
        
        Returns:
            Diccionario con todos los valores del estado actual
        """
        return {
            "letra_actual": self.letra_actual,
            "palabra_actual": self.palabra_actual,
            "frase_completa": self.frase_completa,
            "confianza": self.confianza_actual,
            "pausa_detectada": self.pausa_detectada,
            "frase_para_corregir": self.frase_para_corregir
        }

    def limpiar_todo(self) -> None:
        """
        Reinicia todo el estado del reconocedor.
        Útil cuando el usuario presiona el botón "Limpiar".
        """
        self.letra_actual = ""
        self.palabra_actual = ""
        self.frase_completa = ""
        self.confianza_actual = 0.0
        self.ultimo_gesto_nombre = None
        self.gesto_confirmado = False
        self.pausa_detectada = False
        self.frase_para_corregir = ""
        self.buffer_gestos.clear()
        print("[Reconocedor] Estado limpiado")

    def consumir_pausa(self) -> Optional[str]:
        """
        Consume la pausa detectada y retorna la frase para corregir.
        Después de llamar esto, pausa_detectada queda en False.
        
        Returns:
            La frase lista para corregir, o None si no hay pausa
        """
        if self.pausa_detectada:
            frase = self.frase_para_corregir
            self.pausa_detectada = False    # Marcar como consumida
            self.frase_para_corregir = ""   # Limpiar frase pendiente
            return frase
        return None

    def forzar_fin_palabra(self) -> None:
        """
        Fuerza el final de la palabra actual.
        Útil para el botón manual de separar palabras.
        """
        if self.palabra_actual:
            if self.frase_completa:
                self.frase_completa += " " + self.palabra_actual
            else:
                self.frase_completa = self.palabra_actual
            self.palabra_actual = ""
            self.pausa_detectada = True
            self.frase_para_corregir = self.frase_completa
