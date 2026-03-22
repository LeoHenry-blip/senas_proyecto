"""
base_datos.py
=============
Módulo para cargar y gestionar la base de datos de gestos.
Compara gestos en tiempo real usando distancia euclidiana.
"""

import json       # Para leer el archivo JSON de gestos
import numpy as np  # Para operaciones matemáticas vectoriales
import os         # Para manejo de rutas de archivos
from typing import Optional, Tuple  # Tipos para anotaciones


class BaseDatosGestos:
    """
    Clase que carga la base de datos de gestos y realiza comparaciones.
    Usa distancia euclidiana para encontrar el gesto más parecido.
    """

    def __init__(self, ruta_json: str = "gestos.json"):
        """
        Inicializa la base de datos cargando el archivo JSON.
        
        Args:
            ruta_json: Ruta al archivo gestos.json
        """
        # Diccionario donde se guardan los vectores de cada gesto
        self.gestos: dict = {}
        
        # Umbral de similitud: si la distancia supera este valor, se ignora
        self.umbral_similitud: float = 0.35
        
        # Cargar los gestos desde el archivo JSON
        self._cargar_gestos(ruta_json)
        
        print(f"[BaseDatos] Gestos cargados: {len(self.gestos)}")

    def _cargar_gestos(self, ruta: str) -> None:
        """
        Lee el archivo JSON y convierte los landmarks en vectores numpy.
        
        Args:
            ruta: Ruta al archivo JSON
        """
        # Verificar que el archivo existe
        if not os.path.exists(ruta):
            print(f"[BaseDatos] ADVERTENCIA: No se encontró '{ruta}'")
            print("[BaseDatos] Usando base de datos vacía.")
            return

        try:
            # Abrir y leer el archivo JSON
            with open(ruta, 'r', encoding='utf-8') as archivo:
                datos = json.load(archivo)

            # Iterar sobre cada gesto en la base de datos
            for nombre, info in datos.get("gestures", {}).items():
                # Convertir la lista de coordenadas a un array numpy
                vector = np.array(info["landmarks"], dtype=np.float32)
                
                # Normalizar el vector para que tenga longitud 1
                # Esto hace la comparación más robusta a escala
                norma = np.linalg.norm(vector)
                if norma > 0:
                    vector = vector / norma
                
                # Guardar: nombre del gesto -> vector normalizado
                self.gestos[nombre] = {
                    "vector": vector,
                    "tipo": info.get("type", "unknown"),
                    "descripcion": info.get("description", "")
                }

        except json.JSONDecodeError as e:
            print(f"[BaseDatos] Error al leer JSON: {e}")
        except Exception as e:
            print(f"[BaseDatos] Error inesperado: {e}")

    def extraer_vector_mano(self, landmarks) -> Optional[np.ndarray]:
        """
        Convierte los landmarks de MediaPipe a un vector numpy normalizado.
        
        Args:
            landmarks: Objeto landmarks de MediaPipe con 21 puntos
            
        Returns:
            Vector numpy normalizado o None si hay error
        """
        try:
            # Extraer las 21 coordenadas (x, y, z) de cada punto
            puntos = []
            for punto in landmarks.landmark:
                puntos.extend([punto.x, punto.y, punto.z])
            
            # Convertir a array numpy
            vector = np.array(puntos, dtype=np.float32)
            
            # --- Normalización ---
            # Restar el punto de la muñeca (primer punto, índice 0-2)
            # para hacer el gesto independiente de su posición en pantalla
            muñeca_x = vector[0]  # Coordenada X de la muñeca
            muñeca_y = vector[1]  # Coordenada Y de la muñeca
            
            # Restar la posición de la muñeca de todos los puntos X e Y
            for i in range(21):
                vector[i * 3] -= muñeca_x      # Normalizar X
                vector[i * 3 + 1] -= muñeca_y  # Normalizar Y
                # Z se deja sin cambios
            
            # Normalizar el vector a longitud unitaria
            norma = np.linalg.norm(vector)
            if norma > 0:
                vector = vector / norma
            
            return vector
            
        except Exception as e:
            print(f"[BaseDatos] Error extrayendo vector: {e}")
            return None

    def buscar_gesto(self, vector_query: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Busca el gesto más parecido al vector dado.
        Usa distancia euclidiana entre vectores normalizados.
        
        Args:
            vector_query: Vector del gesto capturado en tiempo real
            
        Returns:
            Tupla (nombre_gesto, confianza) donde confianza va de 0 a 1
            Retorna (None, 0.0) si no hay coincidencia suficiente
        """
        # Si no hay gestos cargados, retornar vacío
        if not self.gestos:
            return None, 0.0
        
        # Variable para guardar la mejor coincidencia
        mejor_nombre = None
        menor_distancia = float('inf')  # Iniciar con infinito
        
        # Comparar contra cada gesto en la base de datos
        for nombre, datos in self.gestos.items():
            vector_referencia = datos["vector"]
            
            # Calcular distancia euclidiana entre los dos vectores
            # Menor distancia = más parecidos
            distancia = np.linalg.norm(vector_query - vector_referencia)
            
            # Actualizar si esta es la menor distancia encontrada
            if distancia < menor_distancia:
                menor_distancia = distancia
                mejor_nombre = nombre
        
        # Si la distancia supera el umbral, no hay coincidencia confiable
        if menor_distancia > self.umbral_similitud:
            return None, 0.0
        
        # Convertir distancia a confianza (0 a 1)
        # Distancia 0 = confianza 1.0, distancia = umbral = confianza 0.0
        confianza = 1.0 - (menor_distancia / self.umbral_similitud)
        confianza = max(0.0, min(1.0, confianza))  # Limitar entre 0 y 1
        
        return mejor_nombre, confianza

    def agregar_gesto(self, nombre: str, landmarks, tipo: str = "custom") -> bool:
        """
        Agrega un nuevo gesto a la base de datos en tiempo de ejecución.
        
        Args:
            nombre: Nombre del nuevo gesto
            landmarks: Landmarks de MediaPipe
            tipo: Tipo de gesto ('letter', 'word', 'custom')
            
        Returns:
            True si se agregó correctamente
        """
        # Extraer y normalizar el vector del gesto
        vector = self.extraer_vector_mano(landmarks)
        if vector is None:
            return False
        
        # Guardar en el diccionario de gestos
        self.gestos[nombre] = {
            "vector": vector,
            "tipo": tipo,
            "descripcion": f"Gesto personalizado: {nombre}"
        }
        
        print(f"[BaseDatos] Gesto '{nombre}' agregado exitosamente")
        return True

    def obtener_estadisticas(self) -> dict:
        """
        Retorna estadísticas de la base de datos cargada.
        
        Returns:
            Diccionario con estadísticas
        """
        # Contar por tipo
        letras = sum(1 for g in self.gestos.values() if g["tipo"] == "letter")
        palabras = sum(1 for g in self.gestos.values() if g["tipo"] == "word")
        personalizados = sum(1 for g in self.gestos.values() if g["tipo"] == "custom")
        
        return {
            "total": len(self.gestos),
            "letras": letras,
            "palabras": palabras,
            "personalizados": personalizados,
            "umbral": self.umbral_similitud
        }
