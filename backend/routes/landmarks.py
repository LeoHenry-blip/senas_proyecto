"""
routes/landmarks.py
===================
Endpoint especial para el panel admin:
Recibe un frame JPEG en base64 desde el navegador,
extrae los landmarks de MediaPipe y los retorna como JSON.
Esto permite que el entrenador web funcione sin instalar nada extra.
"""

import base64
import numpy as np
import cv2
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Optional, List

from core.auth import requerir_admin
from detector_manos import DetectorManos

router = APIRouter(prefix="/admin", tags=["Admin - Landmarks"])

# Instancia compartida del detector (se reutiliza entre requests)
_detector = DetectorManos(max_manos=1, confianza_deteccion=0.6)


class FrameBody(BaseModel):
    frame_b64: str   # JPEG codificado en base64


@router.post("/extraer-landmarks")
async def extraer_landmarks(body: FrameBody, admin = Depends(requerir_admin)):
    """
    Recibe un frame JPEG en base64, detecta la mano con MediaPipe
    y retorna el vector de 63 landmarks normalizado.

    Returns:
        {"landmarks": [f1, f2, ..., f63]} si hay mano
        {"landmarks": null, "error": "..."} si no hay mano
    """
    try:
        # Decodificar base64 a bytes
        jpeg_bytes = base64.b64decode(body.frame_b64)

        # Decodificar JPEG a numpy array BGR
        np_arr = np.frombuffer(jpeg_bytes, np.uint8)
        frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return {"landmarks": None, "error": "Frame inválido"}

        # Detectar manos
        _, resultados = _detector.detectar(frame)

        if not _detector.hay_manos(resultados):
            return {"landmarks": None, "error": "No se detectó mano"}

        # Extraer landmarks de la primera mano
        landmarks = _detector.obtener_primera_mano(resultados)

        # Extraer vector de coordenadas
        puntos = []
        for punto in landmarks.landmark:
            puntos.extend([punto.x, punto.y, punto.z])

        vector = np.array(puntos, dtype=np.float32)

        # Normalizar restando muñeca
        muñeca_x, muñeca_y = vector[0], vector[1]
        for i in range(21):
            vector[i*3]   -= muñeca_x
            vector[i*3+1] -= muñeca_y

        # Normalizar a longitud unitaria
        norma = np.linalg.norm(vector)
        if norma > 0:
            vector = vector / norma

        return {
            "landmarks": vector.tolist(),
            "num_puntos": 21,
            "detectado": True
        }

    except Exception as e:
        return {"landmarks": None, "error": str(e)}
