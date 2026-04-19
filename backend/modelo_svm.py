"""
modelo_svm.py  v2.0
====================
Motor SVM para reconocimiento de gestos con movimiento.

NOVEDAD v2.0 — Soporte de DOS MANOS:
  Cada frame ahora es un vector de 126 floats en vez de 63:
    [0..62]   → mano derecha normalizada
    [63..125] → mano izquierda normalizada (ceros si no hay)

Features extraídas (D = 126 dims):
  - Posición media        D floats
  - Std del movimiento    D floats
  - Delta inicio→fin      D floats
  - Velocidad media       D floats
  - Velocidad máxima      D floats
  Total: 630 features por muestra

Compatibilidad legacy:
  Si llega una secuencia con shape (N, 63), la mitad izquierda
  se rellena con ceros automáticamente. Así los modelos viejos
  siguen prediciendo (con menor precisión) hasta que se re-entrene.
"""

import os
import pickle
import numpy as np
from typing import Optional, Tuple, List, Dict

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold

RUTA_MODELO     = os.path.join(os.path.dirname(__file__), "modelo_svm.pkl")
FRAMES_ESTANDAR = 30   # Frames a los que se resamplea cada secuencia
DIMS_POR_FRAME  = 126  # 63 mano derecha + 63 mano izquierda


# =============================================================================
# UTILIDADES
# =============================================================================

def resamplear(frames: np.ndarray, target: int) -> np.ndarray:
    """Resamplea (N, D) → (target, D) por interpolación lineal."""
    n = len(frames)
    if n == target:
        return frames
    if n == 1:
        return np.tile(frames, (target, 1))
    indices = np.linspace(0, n - 1, target)
    lo = np.floor(indices).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    t  = (indices - lo)[:, None]
    return frames[lo] * (1 - t) + frames[hi] * t


def _a_126(arr: np.ndarray) -> np.ndarray:
    """
    Garantiza que el array tenga shape (N, 126).
    Si viene con 63, rellena la parte izquierda con ceros (modo legacy).
    """
    if arr.shape[1] == 126:
        return arr
    if arr.shape[1] == 63:
        return np.concatenate(
            [arr, np.zeros((len(arr), 63), dtype=np.float32)], axis=1
        )
    raise ValueError(f"Se esperan 63 o 126 dims por frame, hay {arr.shape[1]}")


def extraer_features(secuencia: np.ndarray) -> np.ndarray:
    """
    Extrae un vector de features fijo (D×5) desde una secuencia de frames.

    Args:
        secuencia: Array (N, D) — D puede ser 63 o 126

    Returns:
        Vector 1D de D×5 features
    """
    arr = _a_126(np.array(secuencia, dtype=np.float32))
    seq = resamplear(arr, FRAMES_ESTANDAR)   # (30, 126)

    pos_media = np.mean(seq, axis=0)         # (126,)
    desv      = np.std(seq,  axis=0)         # (126,)
    delta     = seq[-1] - seq[0]             # (126,)
    diffs     = np.diff(seq, axis=0)         # (29, 126)
    vel_media = np.mean(np.abs(diffs), axis=0)
    vel_max   = np.max(np.abs(diffs),  axis=0)

    return np.concatenate(
        [pos_media, desv, delta, vel_media, vel_max]
    ).astype(np.float32)   # (630,)


# =============================================================================
# MODELO SVM
# =============================================================================

class ModeloSVM:
    """Clasificador SVM para gestos con movimiento de 1 o 2 manos."""

    def __init__(self):
        self.pipeline:  Optional[Pipeline] = None
        self.clases:    List[str] = []
        self.entrenado: bool = False
        self.dims:      int  = DIMS_POR_FRAME

    # ── Carga / guardado ─────────────────────────────────────────────

    def cargar(self, ruta: str = RUTA_MODELO) -> bool:
        if not os.path.exists(ruta):
            print(f"[SVM] Modelo no encontrado en {ruta}. Entrena primero.")
            return False
        try:
            with open(ruta, 'rb') as f:
                datos = pickle.load(f)
            self.pipeline  = datos["pipeline"]
            self.clases    = datos["clases"]
            self.dims      = datos.get("dims", 63)
            self.entrenado = True
            print(f"[SVM] Modelo cargado: {len(self.clases)} gestos, "
                  f"dims={self.dims} → {self.clases}")
            return True
        except Exception as e:
            print(f"[SVM] Error cargando: {e}")
            return False

    def guardar(self, ruta: str = RUTA_MODELO) -> bool:
        if not self.entrenado:
            return False
        try:
            with open(ruta, 'wb') as f:
                pickle.dump({
                    "pipeline": self.pipeline,
                    "clases":   self.clases,
                    "dims":     self.dims,
                }, f)
            print(f"[SVM] Guardado en {ruta} (dims={self.dims})")
            return True
        except Exception as e:
            print(f"[SVM] Error guardando: {e}")
            return False

    # ── Entrenamiento ─────────────────────────────────────────────────

    def entrenar(self, muestras: Dict[str, List[np.ndarray]]) -> dict:
        """
        Entrena el SVM.

        Args:
            muestras: {nombre: [array(N,63), array(N,126), ...]}
                      Las secuencias pueden mezclar dims — se normalizan a 126.

        Returns:
            {"ok": True, "precision": 0.93, ...}
        """
        X, y = [], []

        for nombre, seqs in muestras.items():
            if len(seqs) < 3:
                print(f"[SVM] '{nombre}': solo {len(seqs)} muestras, se omite")
                continue
            for seq in seqs:
                arr = np.array(seq, dtype=np.float32)
                if arr.ndim != 2 or len(arr) < 5:
                    continue
                if arr.shape[1] not in (63, 126):
                    print(f"[SVM] Shape inválido {arr.shape}, omitido")
                    continue
                X.append(extraer_features(arr))
                y.append(nombre)

        clases = sorted(set(y))
        if len(clases) < 2:
            return {"ok": False,
                    "error": f"Se necesitan ≥2 gestos. Solo: {clases}"}

        X = np.array(X)
        y = np.array(y)

        self.dims = DIMS_POR_FRAME
        print(f"[SVM] Entrenando: {len(X)} muestras, "
              f"{len(clases)} gestos, {X.shape[1]} features...")

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svm",    SVC(
                kernel       = "rbf",
                C            = 10.0,
                gamma        = "scale",
                probability  = True,
                class_weight = "balanced",
            ))
        ])
        self.pipeline.fit(X, y)
        self.clases    = clases
        self.entrenado = True

        # Validación cruzada
        precision   = 0.0
        min_samples = min(int(np.sum(y == c)) for c in clases)
        n_splits    = min(5, min_samples)

        if n_splits >= 2:
            cv     = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            scores = cross_val_score(self.pipeline, X, y, cv=cv)
            precision = float(np.mean(scores))
            print(f"[SVM] CV {n_splits}-fold: {precision:.1%} ±{np.std(scores):.1%}")
        else:
            preds     = self.pipeline.predict(X)
            precision = float(np.mean(preds == y))
            print(f"[SVM] Precisión entrenamiento: {precision:.1%}")

        conteo = {c: int(np.sum(y == c)) for c in clases}
        return {
            "ok":                 True,
            "precision":          round(precision, 4),
            "total_muestras":     len(X),
            "gestos":             self.clases,
            "muestras_por_gesto": conteo,
            "dims_por_frame":     self.dims,
            "features_totales":   X.shape[1],
        }

    # ── Predicción ───────────────────────────────────────────────────

    def predecir(self, secuencia: np.ndarray,
                 umbral: float = 0.45) -> Tuple[Optional[str], float]:
        """
        Predice el gesto de una secuencia.

        Args:
            secuencia: Array (N, 63) o (N, 126)
            umbral:    Confianza mínima para aceptar

        Returns:
            (nombre, confianza) o (None, confianza)
        """
        if not self.entrenado or self.pipeline is None:
            return None, 0.0
        if len(secuencia) < 5:
            return None, 0.0
        try:
            features  = extraer_features(secuencia).reshape(1, -1)
            probs     = self.pipeline.predict_proba(features)[0]
            idx       = int(np.argmax(probs))
            confianza = float(probs[idx])
            nombre    = self.pipeline.classes_[idx]
            if confianza < umbral:
                return None, confianza
            return nombre, confianza
        except Exception as e:
            print(f"[SVM] Error predecir: {e}")
            return None, 0.0

    def predecir_ranking(self, secuencia: np.ndarray) -> List[dict]:
        """Ranking completo de todos los gestos."""
        if not self.entrenado or self.pipeline is None:
            return []
        if len(secuencia) < 5:
            return []
        try:
            features = extraer_features(secuencia).reshape(1, -1)
            probs    = self.pipeline.predict_proba(features)[0]
            ranking  = [
                {"nombre": str(c), "confianza": round(float(p), 4)}
                for c, p in zip(self.pipeline.classes_, probs)
            ]
            ranking.sort(key=lambda x: x["confianza"], reverse=True)
            return ranking
        except Exception as e:
            print(f"[SVM] Error ranking: {e}")
            return []


# =============================================================================
# SINGLETON GLOBAL
# =============================================================================

_modelo_global: Optional[ModeloSVM] = None


def obtener_modelo() -> ModeloSVM:
    global _modelo_global
    if _modelo_global is None:
        _modelo_global = ModeloSVM()
        _modelo_global.cargar()
    return _modelo_global


def recargar_modelo() -> bool:
    global _modelo_global
    _modelo_global = ModeloSVM()
    return _modelo_global.cargar()