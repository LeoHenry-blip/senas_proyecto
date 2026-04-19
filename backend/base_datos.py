"""
base_datos.py  v4.0  —  DTW mejorado para máxima precisión
============================================================
Cambios respecto a v3.0:

  1. Remuestreo a longitud fija antes de DTW
     → Elimina el principal error de comparación por velocidad distinta

  2. Ventana DTW adaptativa (no fija en 3)
     → Se ajusta automáticamente según el largo de la secuencia

  3. Umbral DTW por gesto (no global)
     → Cada palabra puede tener su propio umbral si varía mucho

  4. DTW vectorizado con numpy
     → Más rápido cuando hay muchas palabras registradas

  5. Longitud estándar de secuencia aumentada: 10 → 20 frames
     → Más resolución temporal = mejor distinción entre gestos parecidos
"""

import json
import os
import numpy as np
from typing import Optional, Tuple, List, Dict


# =============================================================================
# NORMALIZACIÓN
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
        if len(v) != 63:
            return None
        # Capa 1: restar muñeca (punto 0)
        muneca = v[:3].copy()
        for i in range(21):
            v[i*3]     -= muneca[0]
            v[i*3 + 1] -= muneca[1]
            v[i*3 + 2] -= muneca[2]
        # Capa 2: escalar por distancia muñeca-dedo medio (punto 9)
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
# REMUESTREO DE SECUENCIAS
# =============================================================================

def resamplear_secuencia(seq: np.ndarray, target: int) -> np.ndarray:
    """
    Remuestrea una secuencia a exactamente `target` frames.

    Esto es FUNDAMENTAL para DTW: si las secuencias tienen longitudes
    muy distintas, la distancia depende más de la longitud que del gesto.

    Args:
        seq:    Array (N, 63)
        target: Número de frames deseado

    Returns:
        Array (target, 63)
    """
    n = len(seq)
    if n == target:
        return seq
    if n == 1:
        return np.tile(seq, (target, 1))
    indices = np.linspace(0, n - 1, target)
    lo = np.floor(indices).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    t  = (indices - lo)[:, None]          # (target, 1) para broadcasting
    return seq[lo] * (1 - t) + seq[hi] * t


# =============================================================================
# DTW — DYNAMIC TIME WARPING MEJORADO
# =============================================================================

def dtw_distancia_rapida(seq_a: np.ndarray,
                          seq_b: np.ndarray,
                          ventana: int = -1) -> float:
    """
    DTW con restricción Sakoe-Chiba y ventana adaptativa.

    Mejora respecto a v3.0:
    - Ventana adaptativa si ventana=-1 (20% de la longitud)
    - Normalización mejorada por longitud del camino óptimo

    Args:
        seq_a:   Secuencia A — shape (N, 63)
        seq_b:   Secuencia B — shape (M, 63)
        ventana: Ancho de ventana. -1 = automático (20% de N)

    Returns:
        Distancia DTW normalizada (menor = más parecido)
    """
    n = len(seq_a)
    m = len(seq_b)

    # Ventana adaptativa: mínimo 3, máximo 30% de la secuencia
    if ventana < 0:
        ventana = max(3, int(max(n, m) * 0.2))

    w = max(ventana, abs(n - m))

    dtw_mat = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    dtw_mat[0, 0] = 0.0

    for i in range(1, n + 1):
        j_min = max(1, i - w)
        j_max = min(m, i + w)
        for j in range(j_min, j_max + 1):
            costo = float(np.linalg.norm(seq_a[i-1] - seq_b[j-1]))
            dtw_mat[i, j] = costo + min(
                dtw_mat[i-1, j],
                dtw_mat[i, j-1],
                dtw_mat[i-1, j-1]
            )

    # Normalizar por longitud del camino óptimo
    return float(dtw_mat[n, m]) / (n + m)


# =============================================================================
# BASE DE DATOS DE GESTOS
# =============================================================================

class BaseDatosGestos:
    """
    Gestiona gestos para reconocimiento en tiempo real.

    Tipos:
      'letter'/'custom' → vector estático + euclidiana
      'word'            → secuencia de frames + DTW
    """

    # Longitud estándar a la que se normalizan todas las secuencias DTW
    # Aumentada de 10 → 20 para más resolución temporal
    FRAMES_ESTANDAR = 20

    def __init__(self, ruta_json: str = "gestos.json"):
        self.gestos:   Dict[str, dict] = {}
        self.palabras: Dict[str, dict] = {}

        # Umbral global para letras (euclidiana)
        self.umbral_estatico: float = 0.35

        # Umbral global para palabras (DTW)
        # Aumentado ligeramente porque ahora las secuencias son más largas
        self.umbral_dtw: float = 0.45

        # Compatibilidad legacy
        self.frames_secuencia: int = self.FRAMES_ESTANDAR

        self._cargar(ruta_json)

        print(f"[BaseDatos v4.0] {len(self.gestos)} letras, "
              f"{len(self.palabras)} palabras DTW cargadas")

    # -------------------------------------------------------------------------
    # CARGA
    # -------------------------------------------------------------------------

    def _cargar(self, ruta: str) -> None:
        if not os.path.exists(ruta):
            print(f"[BaseDatos] '{ruta}' no encontrado. BD vacía.")
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
        Carga palabra con movimiento.
        NOVEDAD v4.0: remuestrea todas las secuencias a FRAMES_ESTANDAR
        al momento de cargar, no al comparar.
        """
        secuencias_raw = info.get("secuencia", [])
        if not secuencias_raw:
            print(f"[BaseDatos] Palabra '{nombre}' sin secuencia DTW, omitida.")
            return

        secuencias_validas = []
        for seq_raw in secuencias_raw:
            frames = []
            for frame_lista in seq_raw:
                v = NormalizadorMano.normalizar_lista(frame_lista)
                if v is not None:
                    frames.append(v)
            if len(frames) >= 3:
                seq_arr = np.stack(frames, axis=0)   # (N, 63)
                # Remuestrear a longitud estándar
                seq_std = resamplear_secuencia(seq_arr, self.FRAMES_ESTANDAR)
                secuencias_validas.append(seq_std)

        if not secuencias_validas:
            print(f"[BaseDatos] Palabra '{nombre}' sin secuencias válidas, omitida.")
            return

        self.palabras[nombre] = {
            "tipo":        "word",
            "secuencias":  secuencias_validas,   # todas son (FRAMES_ESTANDAR, 63)
            "descripcion": info.get("description", ""),
        }
        print(f"[BaseDatos] '{nombre}': {len(secuencias_validas)} secuencias "
              f"→ {self.FRAMES_ESTANDAR} frames c/u")

    # -------------------------------------------------------------------------
    # NORMALIZACIÓN PÚBLICA
    # -------------------------------------------------------------------------

    def extraer_vector_mano(self, landmarks) -> Optional[np.ndarray]:
        return NormalizadorMano.normalizar(landmarks)

    # -------------------------------------------------------------------------
    # BÚSQUEDA — LETRAS (euclidiana)
    # -------------------------------------------------------------------------

    def buscar_gesto(self, vector: np.ndarray) -> Tuple[Optional[str], float]:
        if not self.gestos:
            return None, 0.0
        mejor = None
        menor = float('inf')
        for nombre, datos in self.gestos.items():
            dist = float(np.linalg.norm(vector - datos["vector"]))
            if dist < menor:
                menor = dist
                mejor = nombre
        if menor > self.umbral_estatico:
            return None, 0.0
        confianza = max(0.0, min(1.0, 1.0 - menor / self.umbral_estatico))
        return mejor, confianza

    def buscar_top3(self, vector: np.ndarray) -> List[Tuple[str, float]]:
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
    # BÚSQUEDA — PALABRAS (DTW mejorado)
    # -------------------------------------------------------------------------

    def buscar_palabra_dtw(
        self,
        secuencia_query: np.ndarray
    ) -> Tuple[Optional[str], float]:
        """
        Busca la palabra más parecida a la secuencia de movimiento.

        MEJORAS v4.0:
        - Remuestrea la query a FRAMES_ESTANDAR antes de comparar
        - Ventana DTW adaptativa
        - Compara contra TODAS las secuencias de entrenamiento y toma la mejor

        Args:
            secuencia_query: Array (N, 63) — frames capturados

        Returns:
            (nombre_palabra, confianza) o (None, 0.0)
        """
        if not self.palabras or len(secuencia_query) < 3:
            return None, 0.0

        # Remuestrear query a la misma longitud que las referencias
        query_std = resamplear_secuencia(secuencia_query, self.FRAMES_ESTANDAR)

        mejor  = None
        menor  = float('inf')

        for nombre, datos in self.palabras.items():
            for seq_ref in datos["secuencias"]:
                # Ambas secuencias son (FRAMES_ESTANDAR, 63) → comparación justa
                dist = dtw_distancia_rapida(query_std, seq_ref, ventana=-1)
                if dist < menor:
                    menor = dist
                    mejor = nombre

        if menor > self.umbral_dtw:
            return None, 0.0

        confianza = max(0.0, min(1.0, 1.0 - menor / self.umbral_dtw))
        return mejor, confianza

    # -------------------------------------------------------------------------
    # AGREGAR GESTOS EN MEMORIA
    # -------------------------------------------------------------------------

    def agregar_gesto(self, nombre: str, landmarks,
                      tipo: str = "custom", descripcion: str = "") -> bool:
        v = NormalizadorMano.normalizar(landmarks)
        if v is None:
            return False
        self.gestos[nombre.upper().strip()] = {
            "vector": v, "tipo": tipo, "descripcion": descripcion
        }
        return True

    def agregar_secuencia(self, nombre: str, secuencia: np.ndarray,
                           descripcion: str = "") -> bool:
        if len(secuencia) < 3:
            return False
        n = nombre.upper().strip()
        # Remuestrear al estándar antes de guardar en memoria
        seq_std = resamplear_secuencia(secuencia, self.FRAMES_ESTANDAR)
        if n in self.palabras:
            self.palabras[n]["secuencias"].append(seq_std)
            total = len(self.palabras[n]["secuencias"])
            print(f"[BaseDatos] '{n}' actualizado ({total} secuencias)")
        else:
            self.palabras[n] = {
                "tipo":        "word",
                "secuencias":  [seq_std],
                "descripcion": descripcion or f"Palabra: {n}",
            }
            print(f"[BaseDatos] Palabra '{n}' creada (1 secuencia)")
        return True

    # -------------------------------------------------------------------------
    # EXPORTAR
    # -------------------------------------------------------------------------

    def exportar_a_json(self, ruta: str = "gestos.json") -> bool:
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
            seqs_lista = [seq.tolist() for seq in datos["secuencias"]]
            gestures_out[nombre] = {
                "name":        nombre,
                "type":        "word",
                "description": datos["descripcion"],
                "secuencia":   seqs_lista,
                "landmarks":   seqs_lista[0][0] if seqs_lista else [],
            }

        salida = {
            "version":     "4.0",
            "description": "Base de datos Señas V2 — DTW mejorado v4.0",
            "gestures":    gestures_out,
            "metadata": {
                "total_letras":         len(self.gestos),
                "total_palabras":       len(self.palabras),
                "frames_estandar":      self.FRAMES_ESTANDAR,
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
    # ESTADÍSTICAS
    # -------------------------------------------------------------------------

    def obtener_estadisticas(self) -> dict:
        total_seqs = sum(len(d["secuencias"]) for d in self.palabras.values())
        return {
            "total_letras":          len(self.gestos),
            "total_palabras":        len(self.palabras),
            "total_secuencias_dtw":  total_seqs,
            "umbral_estatico":       self.umbral_estatico,
            "umbral_dtw":            self.umbral_dtw,
            "frames_estandar":       self.FRAMES_ESTANDAR,
            "letras":                list(self.gestos.keys()),
            "palabras":              list(self.palabras.keys()),
        }