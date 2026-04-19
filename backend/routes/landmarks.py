"""
routes/landmarks.py  v3.0
==========================
Endpoint para extraer landmarks desde el navegador.

NOVEDAD v3.0 — Soporte de DOS MANOS:
  Retorna un vector de 126 floats (63 por mano).
  Si solo hay una mano visible, la mitad faltante se rellena con ceros
  para mantener siempre la misma dimensión.

  Orden del vector de salida:
    [0..62]   → mano dominante (la más a la derecha en imagen espejada)
    [63..125] → mano secundaria (la más a la izquierda)

  Si solo hay 1 mano detectada:
    - Se identifica si es derecha o izquierda usando handedness de MediaPipe
    - La mano ausente se rellena con np.zeros(63)

  Normalización: las mismas 3 capas de NormalizadorMano, aplicadas
  independientemente a cada mano antes de concatenar.
"""

import base64
import numpy as np
import cv2
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from core.auth import requerir_admin
from detector_manos import DetectorManos

router = APIRouter(prefix="/admin", tags=["Admin - Landmarks"])

# max_manos=2 para capturar ambas manos
_detector = DetectorManos(max_manos=2, confianza_deteccion=0.6)

VECTOR_VACIO = np.zeros(63, dtype=np.float32)


class FrameBody(BaseModel):
    frame_b64: str


def _normalizar_63(puntos_raw: list) -> np.ndarray:
    """
    Aplica normalización 3 capas a una lista de 63 floats crudos.
    Igual que NormalizadorMano._aplicar() en base_datos.py.
    """
    v = np.array(puntos_raw, dtype=np.float32)

    # Capa 1: restar muñeca
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


@router.post("/extraer-landmarks")
async def extraer_landmarks(body: FrameBody, admin = Depends(requerir_admin)):
    """
    Detecta hasta 2 manos en el frame y retorna vector de 126 floats.

    Estructura de respuesta:
      landmarks:       [126 floats] — mano_derecha(63) + mano_izquierda(63)
      landmarks_right: [63 floats]  — solo mano derecha (o null)
      landmarks_left:  [63 floats]  — solo mano izquierda (o null)
      manos_detectadas: 0, 1 o 2
      detectado:       true si hay al menos 1 mano
    """
    try:
        jpeg_bytes = base64.b64decode(body.frame_b64)
        np_arr     = np.frombuffer(jpeg_bytes, np.uint8)
        frame      = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return {"landmarks": None, "detectado": False, "error": "Frame inválido"}

        _, resultados = _detector.detectar(frame)

        if not _detector.hay_manos(resultados):
            return {
                "landmarks":        None,
                "landmarks_right":  None,
                "landmarks_left":   None,
                "manos_detectadas": 0,
                "detectado":        False,
            }

        manos      = resultados.multi_hand_landmarks
        handedness = resultados.multi_handedness   # "Left" / "Right" de MediaPipe

        # MediaPipe reporta handedness desde la perspectiva de la cámara.
        # Como el frame llega espejado (scaleX(-1) en el frontend),
        # "Right" de MediaPipe = mano derecha real del usuario.
        vector_derecha   = VECTOR_VACIO.copy()
        vector_izquierda = VECTOR_VACIO.copy()
        tiene_derecha    = False
        tiene_izquierda  = False

        for lm, hand_info in zip(manos, handedness):
            # Clasificación de MediaPipe
            lado = hand_info.classification[0].label  # "Left" o "Right"

            puntos = []
            for punto in lm.landmark:
                puntos.extend([punto.x, punto.y, punto.z])

            v_norm = _normalizar_63(puntos)

            if lado == "Right":
                vector_derecha  = v_norm
                tiene_derecha   = True
            else:
                vector_izquierda = v_norm
                tiene_izquierda  = True

        # Vector completo: derecha primero, izquierda segundo
        vector_126 = np.concatenate([vector_derecha, vector_izquierda])

        return {
            # Vector principal que usan entrenador y SVM
            "landmarks":        vector_126.tolist(),

            # Vectores individuales (útiles para debug y modo letra)
            "landmarks_right":  vector_derecha.tolist()   if tiene_derecha   else None,
            "landmarks_left":   vector_izquierda.tolist() if tiene_izquierda else None,

            "manos_detectadas": (1 if tiene_derecha else 0) + (1 if tiene_izquierda else 0),
            "detectado":        True,
            "num_puntos":       21,
        }

    except Exception as e:
        return {"landmarks": None, "detectado": False, "error": str(e)}