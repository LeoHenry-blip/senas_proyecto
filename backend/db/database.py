"""
base_datos.py  v2.1
====================
Módulo para cargar y gestionar la base de datos de gestos.

Mejora principal vs v1:
  Normalización en 3 capas:
    1. Restar muñeca       → independiente de posición en pantalla
    2. Escalar por mano    → independiente del tamaño de la mano
    3. Vector unitario     → independiente de escala general

Esto hace que el mismo gesto sea reconocido aunque:
  - La mano esté en distinta posición
  - La mano sea más grande o más pequeña
  - La cámara esté más cerca o lejos

Compatible 100% con el gestos.json existente.
"""

import json
import os
import numpy as np
from typing import Optional, Tuple, List


class BaseDatosGestos:
    """
    Carga gestos desde JSON y los compara en tiempo real.
    Usa distancia euclidiana sobre vectores normalizados.
    """

    def __init__(self, ruta_json: str = "gestos.json"):
        """
        Args:
            ruta_json: Ruta al archivo gestos.json
        """
        # {nombre: {"vector": np.ndarray, "tipo": str, "descripcion": str}}
        self.gestos: dict = {}

        # Umbral de similitud: distancia máxima aceptada (0.0 - 1.0)
        # Más bajo = más estricto. 0.35 es un buen balance.
        self.umbral_similitud: float = 0.35

        self._cargar_gestos(ruta_json)
        print(f"[BaseDatos] Gestos cargados: {len(self.gestos)}")

    # =========================================================================
    # CARGA
    # =========================================================================

    def _cargar_gestos(self, ruta: str) -> None:
        """Lee gestos.json y convierte landmarks a vectores numpy normalizados."""
        if not os.path.exists(ruta):
            print(f"[BaseDatos] ADVERTENCIA: '{ruta}' no encontrado. BD vacía.")
            return

        try:
            with open(ruta, 'r', encoding='utf-8') as f:
                datos = json.load(f)

            for nombre, info in datos.get("gestures", {}).items():
                landmarks = info.get("landmarks", [])
                if not landmarks or len(landmarks) != 63:
                    continue

                # Normalizar con las 3 capas
                vector = self._normalizar_lista(landmarks)
                if vector is None:
                    continue

                self.gestos[nombre] = {
                    "vector":     vector,
                    "tipo":       info.get("type",        "unknown"),
                    "descripcion":info.get("description", ""),
                }

        except json.JSONDecodeError as e:
            print(f"[BaseDatos] Error JSON: {e}")
        except Exception as e:
            print(f"[BaseDatos] Error inesperado: {e}")

    # =========================================================================
    # NORMALIZACIÓN  (el cambio clave)
    # =========================================================================

    def _normalizar_lista(self, lista: List[float]) -> Optional[np.ndarray]:
        """
        Normaliza una lista de 63 floats con 3 capas.
        Usado al cargar desde JSON.
        """
        try:
            v = np.array(lista, dtype=np.float32)
            return self._aplicar_normalizacion(v)
        except Exception:
            return None

    def extraer_vector_mano(self, landmarks) -> Optional[np.ndarray]:
        """
        Convierte landmarks de MediaPipe a vector normalizado.
        Llamado en cada frame por el reconocedor.

        Args:
            landmarks: objeto landmarks de MediaPipe (21 puntos)

        Returns:
            Vector numpy de 63 floats normalizado, o None si hay error
        """
        try:
            puntos = []
            for p in landmarks.landmark:
                puntos.extend([p.x, p.y, p.z])
            v = np.array(puntos, dtype=np.float32)
            return self._aplicar_normalizacion(v)
        except Exception as e:
            print(f"[BaseDatos] Error extrayendo vector: {e}")
            return None

    def _aplicar_normalizacion(self, v: np.ndarray) -> Optional[np.ndarray]:
        """
        Aplica las 3 capas de normalización a un vector de 63 floats.

        Capa 1 — Restar muñeca (punto 0):
            Hace el gesto independiente de DÓNDE está la mano en pantalla.
            Antes: mano arriba = coordenadas altas, mano abajo = coordenadas bajas
            Después: siempre relativo a la muñeca = mismo gesto

        Capa 2 — Escalar por distancia muñeca→base dedo medio (punto 9):
            Hace el gesto independiente del TAMAÑO de la mano.
            Antes: mano grande ≠ mano pequeña aunque hagan el mismo gesto
            Después: ambas producen el mismo vector

        Capa 3 — Vector unitario:
            Normalización matemática estándar.
            Garantiza que todos los vectores tienen la misma longitud (=1).
        """
        if len(v) != 63:
            return None

        # ---- Capa 1: restar posición de la muñeca ----
        muñeca = v[:3].copy()   # [x0, y0, z0]
        for i in range(21):
            v[i*3]     -= muñeca[0]
            v[i*3 + 1] -= muñeca[1]
            v[i*3 + 2] -= muñeca[2]

        # ---- Capa 2: escalar por tamaño de la mano ----
        # El punto 9 (base del dedo medio) es un buen indicador del tamaño
        ref_x = v[9 * 3]        # x del punto 9
        ref_y = v[9 * 3 + 1]    # y del punto 9
        distancia_mano = float(np.sqrt(ref_x**2 + ref_y**2))

        if distancia_mano > 1e-6:   # Evitar división por cero
            v = v / distancia_mano

        # ---- Capa 3: normalizar a vector unitario ----
        norma = float(np.linalg.norm(v))
        if norma > 1e-6:
            v = v / norma

        return v

    # =========================================================================
    # BÚSQUEDA
    # =========================================================================

    def buscar_gesto(self, vector_query: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Encuentra el gesto más parecido usando distancia euclidiana.

        Args:
            vector_query: Vector del gesto capturado (ya normalizado)

        Returns:
            (nombre_gesto, confianza_0_a_1)
            (None, 0.0) si no hay coincidencia dentro del umbral
        """
        if not self.gestos:
            return None, 0.0

        mejor_nombre    = None
        menor_distancia = float('inf')

        for nombre, datos in self.gestos.items():
            dist = float(np.linalg.norm(vector_query - datos["vector"]))
            if dist < menor_distancia:
                menor_distancia = dist
                mejor_nombre    = nombre

        # Sin coincidencia suficiente
        if menor_distancia > self.umbral_similitud:
            return None, 0.0

        # Convertir distancia → confianza (0 a 1)
        confianza = max(0.0, min(1.0,
            1.0 - (menor_distancia / self.umbral_similitud)
        ))
        return mejor_nombre, confianza

    def buscar_top3(self, vector_query: np.ndarray) -> List[Tuple[str, float]]:
        """
        Retorna los 3 gestos más cercanos.
        Útil para el panel de prueba.

        Returns:
            Lista de (nombre, confianza) ordenada de mayor a menor confianza
        """
        if not self.gestos:
            return []

        resultados = []
        for nombre, datos in self.gestos.items():
            dist = float(np.linalg.norm(vector_query - datos["vector"]))
            conf = max(0.0, 1.0 - (dist / self.umbral_similitud))
            resultados.append((nombre, conf, dist))

        resultados.sort(key=lambda x: x[2])   # Ordenar por distancia
        return [(n, c) for n, c, _ in resultados[:3]]

    # =========================================================================
    # AGREGAR GESTOS (usado por el entrenador)
    # =========================================================================

    def agregar_gesto(
        self,
        nombre:      str,
        landmarks,
        tipo:        str = "custom",
        descripcion: str = ""
    ) -> bool:
        """
        Agrega un gesto en memoria (no persiste en disco).
        Para persistir, llama a exportar_a_json() después.

        Args:
            nombre:      Nombre del gesto (ej: "A")
            landmarks:   landmarks.landmark de MediaPipe
            tipo:        'letter', 'word' o 'custom'
            descripcion: Texto descriptivo opcional

        Returns:
            True si se agregó correctamente
        """
        vector = self.extraer_vector_mano(landmarks)
        if vector is None:
            return False

        self.gestos[nombre.upper().strip()] = {
            "vector":      vector,
            "tipo":        tipo,
            "descripcion": descripcion or f"Gesto: {nombre}",
        }
        print(f"[BaseDatos] Gesto '{nombre.upper()}' agregado")
        return True

    # =========================================================================
    # ESTADÍSTICAS
    # =========================================================================

    def obtener_estadisticas(self) -> dict:
        """Retorna un resumen de la base de datos cargada."""
        letras  = [n for n, d in self.gestos.items() if d["tipo"] == "letter"]
        palabras= [n for n, d in self.gestos.items() if d["tipo"] == "word"]
        custom  = [n for n, d in self.gestos.items() if d["tipo"] == "custom"]

        return {
            "total":       len(self.gestos),
            "letras":      letras,
            "palabras":    palabras,
            "custom":      custom,
            "umbral":      self.umbral_similitud,
            "estado":      "OK" if self.gestos else "VACÍO",
        }