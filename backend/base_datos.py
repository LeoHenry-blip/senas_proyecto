"""
base_datos.py  v3.0  —  DTW para palabras con movimiento
==========================================================
Módulo de base de datos de gestos con soporte híbrido:

  LETRAS  (tipo='letter' o 'custom'):
    - Un solo vector de 63 floats (posición estática)
    - Comparación por distancia euclidiana (rápido, preciso)

  PALABRAS (tipo='word'):
    - Secuencia de N frames (10 por defecto)
    - Comparación por DTW (Dynamic Time Warping)
    - Robusto a velocidades distintas del movimiento

Formato gestos.json:
  Letras  →  campo "landmarks": [63 floats]
  Palabras → campo "secuencia": [[[63 floats] x 10 frames], ...]
             campo "landmarks": primer frame (compatibilidad)

Normalización en 3 capas para ambos tipos:
  1. Restar muñeca       → independiente de posición
  2. Escalar por mano    → independiente de tamaño
  3. Vector unitario     → independiente de escala
"""

import json
import os
import numpy as np
from typing import Optional, Tuple, List, Dict


# =============================================================================
# NORMALIZADOR
# =============================================================================

class NormalizadorMano:
    """Normalización de landmarks en 3 capas."""

    @staticmethod
    def normalizar(landmarks) -> Optional[np.ndarray]:
        """Convierte landmarks de MediaPipe a vector normalizado."""
        try:
            puntos = []
            for p in landmarks.landmark:
                puntos.extend([p.x, p.y, p.z])
            v = np.array(puntos, dtype=np.float32)
            return NormalizadorMano._aplicar(v)
        except Exception as e:
            print(f"[Normalizador] Error: {e}")
            return None

    @staticmethod
    def normalizar_lista(lista: List[float]) -> Optional[np.ndarray]:
        """Normaliza una lista de 63 floats."""
        try:
            v = np.array(lista, dtype=np.float32)
            if len(v) != 63:
                return None
            return NormalizadorMano._aplicar(v)
        except Exception:
            return None

    @staticmethod
    def _aplicar(v: np.ndarray) -> Optional[np.ndarray]:
        """Aplica las 3 capas de normalización."""
        if len(v) != 63:
            return None

        # Capa 1: restar muneca (punto 0)
        muneca = v[:3].copy()
        for i in range(21):
            v[i*3]     -= muneca[0]
            v[i*3 + 1] -= muneca[1]
            v[i*3 + 2] -= muneca[2]

        # Capa 2: escalar por distancia muneca-dedo medio (punto 9)
        ref_x = v[9 * 3]
        ref_y = v[9 * 3 + 1]
        dist  = float(np.sqrt(ref_x**2 + ref_y**2))
        if dist > 1e-6:
            v = v / dist

        # Capa 3: vector unitario
        norma = float(np.linalg.norm(v))
        if norma > 1e-6:
            v = v / norma

        return v


# =============================================================================
# DTW — DYNAMIC TIME WARPING
# =============================================================================

def dtw_distancia_rapida(seq_a: np.ndarray,
                          seq_b: np.ndarray,
                          ventana: int = 3) -> float:
    """
    DTW con restriccion de ventana Sakoe-Chiba.

    Compara dos secuencias de vectores sin importar diferencias de velocidad.
    La ventana limita el rango de comparacion para mayor eficiencia.

    Args:
        seq_a:   Secuencia A — shape (N, 63)
        seq_b:   Secuencia B — shape (M, 63)
        ventana: Ancho de la ventana de comparacion

    Returns:
        Distancia DTW normalizada (menor = mas parecido)
    """
    n = len(seq_a)
    m = len(seq_b)
    w = max(ventana, abs(n - m))

    # Matriz de costos acumulados
    dtw = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    dtw[0, 0] = 0.0

    for i in range(1, n + 1):
        j_min = max(1, i - w)
        j_max = min(m, i + w)
        for j in range(j_min, j_max + 1):
            costo = float(np.linalg.norm(seq_a[i-1] - seq_b[j-1]))
            dtw[i, j] = costo + min(
                dtw[i-1, j],
                dtw[i, j-1],
                dtw[i-1, j-1]
            )

    # Normalizar por longitud total del camino
    return float(dtw[n, m]) / (n + m)


# =============================================================================
# BASE DE DATOS DE GESTOS
# =============================================================================

class BaseDatosGestos:
    """
    Gestiona gestos para reconocimiento en tiempo real.

    Tipos:
      'letter'/'custom' -> vector estatico + euclidiana
      'word'            -> secuencia de frames + DTW
    """

    def __init__(self, ruta_json: str = "gestos.json"):
        # Letras: {nombre: {"vector": np.ndarray, "tipo": str, "descripcion": str}}
        self.gestos:   Dict[str, dict] = {}

        # Palabras: {nombre: {"secuencias": [np.ndarray], "tipo": str, "descripcion": str}}
        self.palabras: Dict[str, dict] = {}

        # Umbrales
        self.umbral_estatico:  float = 0.35
        self.umbral_dtw:       float = 0.45
        self.frames_secuencia: int   = 10

        self._cargar(ruta_json)

        print(f"[BaseDatos] Gestos cargados: {len(self.gestos)} letras, "
              f"{len(self.palabras)} palabras")

    # -------------------------------------------------------------------------
    # CARGA
    # -------------------------------------------------------------------------

    def _cargar(self, ruta: str) -> None:
        """Carga gestos desde JSON separando letras de palabras."""
        if not os.path.exists(ruta):
            print(f"[BaseDatos] '{ruta}' no encontrado. BD vacia.")
            return

        try:
            with open(ruta, 'r', encoding='utf-8') as f:
                datos = json.load(f)

            for nombre, info in datos.get("gestures", {}).items():
                tipo = info.get("type", "letter")

                if tipo == "word":
                    self._cargar_palabra(nombre, info)
                else:
                    self._cargar_letra(nombre, info, tipo)

        except Exception as e:
            print(f"[BaseDatos] Error cargando JSON: {e}")

    def _cargar_letra(self, nombre: str, info: dict, tipo: str) -> None:
        """Carga un gesto estatico (letra)."""
        landmarks = info.get("landmarks", [])
        if not landmarks or len(landmarks) != 63:
            return
        v = NormalizadorMano.normalizar_lista(landmarks)
        if v is None:
            return
        self.gestos[nombre] = {
            "vector":      v,
            "tipo":        tipo,
            "descripcion": info.get("description", ""),
        }

    def _cargar_palabra(self, nombre: str, info: dict) -> None:
        """
        Carga un gesto de movimiento (palabra) con sus secuencias DTW.
        Formato esperado en JSON:
          "secuencia": [
            [[63 floats] x N frames],   <- secuencia 1 de entrenamiento
            [[63 floats] x N frames],   <- secuencia 2 de entrenamiento
            ...
          ]
        """
        secuencias_raw = info.get("secuencia", [])
        if not secuencias_raw:
            print(f"[BaseDatos] Palabra '{nombre}' sin secuencia DTW, omitida.")
            return

        secuencias_validas = []
        for seq_raw in secuencias_raw:
            # seq_raw es una lista de frames, cada frame es una lista de 63 floats
            frames = []
            for frame_lista in seq_raw:
                v = NormalizadorMano.normalizar_lista(frame_lista)
                if v is not None:
                    frames.append(v)
            if len(frames) >= 3:
                # Convertir a array (N_frames, 63)
                secuencias_validas.append(np.stack(frames, axis=0))

        if not secuencias_validas:
            print(f"[BaseDatos] Palabra '{nombre}' sin secuencias validas, omitida.")
            return

        self.palabras[nombre] = {
            "tipo":        "word",
            "secuencias":  secuencias_validas,
            "descripcion": info.get("description", ""),
        }

    # -------------------------------------------------------------------------
    # NORMALIZACION PUBLICA
    # -------------------------------------------------------------------------

    def extraer_vector_mano(self, landmarks) -> Optional[np.ndarray]:
        """Extrae y normaliza el vector de landmarks de MediaPipe."""
        return NormalizadorMano.normalizar(landmarks)

    # -------------------------------------------------------------------------
    # BUSQUEDA — LETRAS (euclidiana)
    # -------------------------------------------------------------------------

    def buscar_gesto(self, vector: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Busca la letra mas parecida al vector dado.

        Args:
            vector: Vector normalizado de 63 floats

        Returns:
            (nombre, confianza) o (None, 0.0)
        """
        if not self.gestos:
            return None, 0.0

        mejor  = None
        menor  = float('inf')

        for nombre, datos in self.gestos.items():
            dist = float(np.linalg.norm(vector - datos["vector"]))
            if dist < menor:
                menor  = dist
                mejor  = nombre

        if menor > self.umbral_estatico:
            return None, 0.0

        confianza = max(0.0, min(1.0, 1.0 - menor / self.umbral_estatico))
        return mejor, confianza

    def buscar_top3(self, vector: np.ndarray) -> List[Tuple[str, float]]:
        """Top 3 letras mas cercanas. Para panel de prueba."""
        if not self.gestos:
            return []
        res = []
        for nombre, datos in self.gestos.items():
            dist = float(np.linalg.norm(vector - datos["vector"]))
            conf = max(0.0, 1.0 - dist / self.umbral_estatico)
            res.append((nombre, conf, dist))
        res.sort(key=lambda x: x[2])
        return [(n, c) for n, c, _ in res[:3]]

    # -------------------------------------------------------------------------
    # BUSQUEDA — PALABRAS (DTW)
    # -------------------------------------------------------------------------

    def buscar_palabra_dtw(
        self,
        secuencia_query: np.ndarray
    ) -> Tuple[Optional[str], float]:
        """
        Busca la palabra mas parecida a la secuencia de movimiento.
        Compara con DTW contra todas las secuencias de entrenamiento.

        Args:
            secuencia_query: Array (N, 63) — frames capturados del movimiento

        Returns:
            (nombre_palabra, confianza) o (None, 0.0)
        """
        if not self.palabras or len(secuencia_query) < 3:
            return None, 0.0

        mejor  = None
        menor  = float('inf')

        for nombre, datos in self.palabras.items():
            # Comparar contra todas las secuencias de entrenamiento
            for seq_ref in datos["secuencias"]:
                dist = dtw_distancia_rapida(secuencia_query, seq_ref, ventana=3)
                if dist < menor:
                    menor  = dist
                    mejor  = nombre

        if menor > self.umbral_dtw:
            return None, 0.0

        confianza = max(0.0, min(1.0, 1.0 - menor / self.umbral_dtw))
        return mejor, confianza

    # -------------------------------------------------------------------------
    # AGREGAR GESTOS EN MEMORIA
    # -------------------------------------------------------------------------

    def agregar_gesto(self, nombre: str, landmarks,
                      tipo: str = "custom", descripcion: str = "") -> bool:
        """Agrega gesto estatico en memoria."""
        v = NormalizadorMano.normalizar(landmarks)
        if v is None:
            return False
        self.gestos[nombre.upper().strip()] = {
            "vector": v, "tipo": tipo, "descripcion": descripcion
        }
        return True

    def agregar_secuencia(self, nombre: str, secuencia: np.ndarray,
                           descripcion: str = "") -> bool:
        """
        Agrega una secuencia de movimiento (palabra) en memoria.

        Args:
            nombre:    Nombre de la palabra (ej: "HOLA")
            secuencia: Array (N, 63) con los frames normalizados del movimiento
            descripcion: Descripcion opcional

        Returns:
            True si se agrego correctamente
        """
        if len(secuencia) < 3:
            print(f"[BaseDatos] Secuencia muy corta para '{nombre}': {len(secuencia)} frames")
            return False

        n = nombre.upper().strip()

        if n in self.palabras:
            # Anadir secuencia adicional (mas muestras = mas robusto)
            self.palabras[n]["secuencias"].append(secuencia)
            total = len(self.palabras[n]["secuencias"])
            print(f"[BaseDatos] '{n}' actualizado ({total} secuencias)")
        else:
            self.palabras[n] = {
                "tipo":        "word",
                "secuencias":  [secuencia],
                "descripcion": descripcion or f"Palabra: {n}",
            }
            print(f"[BaseDatos] Palabra '{n}' creada (1 secuencia)")

        return True

    # -------------------------------------------------------------------------
    # EXPORTAR
    # -------------------------------------------------------------------------

    def exportar_a_json(self, ruta: str = "gestos.json") -> bool:
        """
        Guarda todos los gestos en gestos.json v3.0.
        Letras: campo 'landmarks'. Palabras: campo 'secuencia'.
        """
        import shutil
        from datetime import datetime

        if os.path.exists(ruta):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            shutil.copy2(ruta, ruta.replace(".json", f"_backup_{ts}.json"))

        gestures_out = {}

        for nombre, datos in self.gestos.items():
            gestures_out[nombre] = {
                "name":        nombre,
                "type":        datos["tipo"],
                "description": datos["descripcion"],
                "landmarks":   datos["vector"].tolist(),
            }

        for nombre, datos in self.palabras.items():
            # Cada secuencia es (N, 63) — convertir a lista de listas
            seqs_lista = [seq.tolist() for seq in datos["secuencias"]]
            gestures_out[nombre] = {
                "name":        nombre,
                "type":        "word",
                "description": datos["descripcion"],
                "secuencia":   seqs_lista,
                # Primer frame del primer entrenamiento como landmark legacy
                "landmarks":   seqs_lista[0][0] if seqs_lista else [],
            }

        salida = {
            "version":     "3.0",
            "description": "Base de datos Senyas V2 — con DTW para palabras",
            "gestures":    gestures_out,
            "metadata": {
                "total_letras":         len(self.gestos),
                "total_palabras":       len(self.palabras),
                "frames_por_secuencia": self.frames_secuencia,
                "exportado_en":         datetime.now().isoformat(),
            }
        }

        try:
            with open(ruta, 'w', encoding='utf-8') as f:
                json.dump(salida, f, ensure_ascii=False, indent=2)
            print(f"[BaseDatos] Exportado: {len(self.gestos)} letras, "
                  f"{len(self.palabras)} palabras → {ruta}")
            return True
        except Exception as e:
            print(f"[BaseDatos] Error exportando: {e}")
            return False

    # -------------------------------------------------------------------------
    # ESTADISTICAS
    # -------------------------------------------------------------------------

    def obtener_estadisticas(self) -> dict:
        total_seqs = sum(len(d["secuencias"]) for d in self.palabras.values())
        return {
            "total_letras":          len(self.gestos),
            "total_palabras":        len(self.palabras),
            "total_secuencias_dtw":  total_seqs,
            "umbral_estatico":       self.umbral_estatico,
            "umbral_dtw":            self.umbral_dtw,
            "frames_secuencia":      self.frames_secuencia,
            "letras":   list(self.gestos.keys()),
            "palabras": list(self.palabras.keys()),
        }
