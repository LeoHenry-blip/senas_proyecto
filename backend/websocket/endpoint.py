"""
websocket/endpoint.py
=====================
Endpoint WebSocket principal: /ws/{sala_id}
"""

import json
import asyncio
import base64
import numpy as np
import cv2
from fastapi import WebSocket, WebSocketDisconnect, Query
from datetime import datetime

from websocket.manager import ws_manager
from db.database import db
from core.auth import decodificar_token
from detector_manos import DetectorManos
from reconocedor import Reconocedor
from ia_corrector import IaCorrector
import os


class SesionReconocimiento:
    def __init__(self, usuario_id: int, nombre: str, sala_id: str):
        self.usuario_id  = usuario_id
        self.nombre      = nombre
        self.sala_id     = sala_id
        self.detector    = DetectorManos(max_manos=1, confianza_deteccion=0.7)
        self.reconocedor = Reconocedor()
        self.corrector   = IaCorrector(
            api_key=os.getenv("IA_API_KEY"),
            usar_api=bool(os.getenv("IA_API_KEY"))
        )
        self.frame_count = 0

    def procesar_frame_jpeg(self, jpeg_bytes: bytes) -> dict:
        self.frame_count += 1
        np_arr = np.frombuffer(jpeg_bytes, np.uint8)
        frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return self.reconocedor.obtener_estado()

        if self.frame_count % 2 == 0:
            _, resultados = self.detector.detectar(frame)
            if self.detector.hay_manos(resultados):
                landmarks = self.detector.obtener_primera_mano(resultados)
                nombre_gesto, confianza = self.reconocedor.procesar_landmarks(landmarks)
                self.reconocedor.actualizar_con_mano(nombre_gesto, confianza)
            else:
                self.reconocedor.actualizar_sin_mano()

        return self.reconocedor.obtener_estado()

    def liberar(self):
        self.detector.liberar()


async def websocket_sala(websocket: WebSocket, sala_id: str,
                          token: str = Query(...)):
    # Autenticar
    payload = decodificar_token(token)
    if not payload:
        await websocket.close(code=4001, reason="Token inválido")
        return

    usuario_id = int(payload["sub"])
    usuario = db.fetchone(
        "SELECT id, nombre, email, rol FROM usuarios WHERE id = %s AND activo = 1",
        (usuario_id,)
    )
    if not usuario:
        await websocket.close(code=4002, reason="Usuario no encontrado")
        return

    reunion = db.fetchone(
        "SELECT id, activa FROM reuniones WHERE codigo = %s", (sala_id,)
    )
    if not reunion or not reunion["activa"]:
        await websocket.close(code=4003, reason="Sala no encontrada o inactiva")
        return

    reunion_db_id  = reunion["id"]
    nombre_usuario = usuario["nombre"]

    cliente = await ws_manager.conectar(websocket, sala_id, usuario_id, nombre_usuario)
    sesion  = SesionReconocimiento(usuario_id, nombre_usuario, sala_id)

    print(f"[WS] '{nombre_usuario}' conectado a sala '{sala_id}'")

    try:
        while True:
            try:
                mensaje_raw = await asyncio.wait_for(websocket.receive(), timeout=30.0)
            except asyncio.TimeoutError:
                try:
                    await websocket.send_json({"tipo": "pong"})
                except Exception:
                    break
                continue

            if "bytes" in mensaje_raw and mensaje_raw["bytes"]:
                jpeg_bytes = mensaje_raw["bytes"]
                await _procesar_frame(sesion, jpeg_bytes, sala_id, usuario_id, nombre_usuario)

            elif "text" in mensaje_raw and mensaje_raw["text"]:
                try:
                    datos = json.loads(mensaje_raw["text"])
                except json.JSONDecodeError:
                    continue
                tipo = datos.get("tipo", "")
                await _procesar_mensaje_json(
                    tipo, datos, sesion, sala_id,
                    usuario_id, nombre_usuario, reunion_db_id
                )

    except WebSocketDisconnect:
        print(f"[WS] '{nombre_usuario}' desconectado")
    except Exception as e:
        print(f"[WS] Error en loop de '{nombre_usuario}': {e}")
    finally:
        sesion.liberar()
        await ws_manager.desconectar(cliente)


async def _procesar_frame(sesion, jpeg_bytes, sala_id, usuario_id, nombre):
    loop   = asyncio.get_event_loop()
    estado = await loop.run_in_executor(None, sesion.procesar_frame_jpeg, jpeg_bytes)

    await ws_manager.enviar_traduccion(
        sala_id, usuario_id, nombre,
        estado.get("letra_actual", ""),
        estado.get("palabra_actual", ""),
        estado.get("frase_completa", ""),
        estado.get("confianza", 0.0)
    )

    if estado.get("pausa_detectada"):
        frase = sesion.reconocedor.consumir_pausa()
        if frase:
            await _procesar_frase_completa(
                frase, sesion, sala_id, usuario_id, nombre,
                sesion.reconocedor.confianza_actual
            )


async def _procesar_mensaje_json(tipo, datos, sesion, sala_id,
                                  usuario_id, nombre, reunion_db_id):
    if tipo == "frame":
        b64 = datos.get("data", "")
        if b64:
            if "," in b64:
                b64 = b64.split(",", 1)[1]
            jpeg_bytes = base64.b64decode(b64)
            await _procesar_frame(sesion, jpeg_bytes, sala_id, usuario_id, nombre)

    elif tipo == "mensaje_texto":
        texto = datos.get("texto", "").strip()
        if not texto:
            return
        msg_id = db.insert(
            "INSERT INTO mensajes (reunion_id, usuario_id, texto_original, tipo) "
            "VALUES (%s, %s, %s, 'texto')",
            (reunion_db_id, usuario_id, texto)
        )
        await ws_manager.enviar_mensaje_chat(
            sala_id, usuario_id, nombre, texto, texto, msg_id
        )

    elif tipo == "fin_frase":
        sesion.reconocedor.forzar_fin_palabra()
        frase = sesion.reconocedor.consumir_pausa()
        if frase:
            await _procesar_frase_completa(
                frase, sesion, sala_id, usuario_id, nombre,
                sesion.reconocedor.confianza_actual
            )

    elif tipo == "limpiar":
        sesion.reconocedor.limpiar_todo()
        await ws_manager.broadcast(sala_id, {
            "tipo": "limpiar_subtitulos", "usuario_id": usuario_id
        })

    elif tipo == "webrtc_señal":
        para_id = datos.get("para")
        señal   = datos.get("señal", {})
        if para_id and señal:
            await ws_manager.enviar_señal_webrtc(sala_id, usuario_id, int(para_id), señal)

    elif tipo == "ping":
        pass


async def _procesar_frase_completa(frase, sesion, sala_id,
                                    usuario_id, nombre, confianza):
    frase_corregida = sesion.corrector.corregir_local(frase)

    reunion = db.fetchone("SELECT id FROM reuniones WHERE codigo = %s", (sala_id,))
    if not reunion:
        return

    msg_id = db.insert(
        "INSERT INTO mensajes "
        "(reunion_id, usuario_id, texto_original, texto_corregido, tipo, confianza) "
        "VALUES (%s, %s, %s, %s, 'senas', %s)",
        (reunion["id"], usuario_id, frase, frase_corregida, confianza)
    )

    await ws_manager.enviar_mensaje_chat(
        sala_id, usuario_id, nombre, frase, frase_corregida, msg_id
    )

    if sesion.corrector.debe_usar_api(frase, confianza):
        async def _actualizar_con_ia():
            loop = asyncio.get_event_loop()
            frase_ia = await loop.run_in_executor(
                None, lambda: sesion.corrector._llamar_api_anthropic(frase)
            )
            if frase_ia and frase_ia != frase_corregida:
                db.execute(
                    "UPDATE mensajes SET texto_corregido = %s WHERE id = %s",
                    (frase_ia, msg_id)
                )
                await ws_manager.broadcast(sala_id, {
                    "tipo":            "mensaje_corregido_ia",
                    "mensaje_id":      msg_id,
                    "texto_corregido": frase_ia,
                })
        asyncio.create_task(_actualizar_con_ia())
