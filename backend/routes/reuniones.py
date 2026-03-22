"""
routes/reuniones.py
===================
Rutas CRUD para reuniones (salas de videollamada + chat).
"""

import random, string
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

from core.auth import get_usuario_actual
from db.database import db

router = APIRouter(prefix="/reuniones", tags=["Reuniones"])


def generar_codigo() -> str:
    """Genera un código único de sala: ej ABC-4X2-QWE"""
    partes = [''.join(random.choices(string.ascii_uppercase + string.digits, k=3)) for _ in range(3)]
    return '-'.join(partes)


class CrearReunionBody(BaseModel):
    nombre: Optional[str] = None


@router.post("/crear", status_code=201)
async def crear_reunion(body: CrearReunionBody, usuario = Depends(get_usuario_actual)):
    """Crea una nueva sala de reunión y retorna su código."""
    # Generar código único (reintentar si hay colisión)
    for _ in range(5):
        codigo = generar_codigo()
        existente = db.fetchone("SELECT id FROM reuniones WHERE codigo = %s", (codigo,))
        if not existente:
            break
    else:
        raise HTTPException(500, "No se pudo generar código único")

    reunion_id = db.insert(
        "INSERT INTO reuniones (codigo, nombre, creador_id) VALUES (%s, %s, %s)",
        (codigo, body.nombre, usuario["id"])
    )

    # Agregar creador como primer participante
    db.execute(
        "INSERT INTO reunion_participantes (reunion_id, usuario_id) VALUES (%s, %s)",
        (reunion_id, usuario["id"])
    )

    return {"ok": True, "codigo": codigo, "reunion_id": reunion_id}


@router.get("/unirse/{codigo}")
async def unirse_reunion(codigo: str, usuario = Depends(get_usuario_actual)):
    """Verifica que la sala existe y registra al usuario como participante."""
    reunion = db.fetchone(
        "SELECT id, nombre, activa FROM reuniones WHERE codigo = %s",
        (codigo.upper(),)
    )
    if not reunion:
        raise HTTPException(404, "Sala no encontrada")
    if not reunion["activa"]:
        raise HTTPException(410, "Esta sala ya fue cerrada")

    # Registrar participación (INSERT IGNORE si ya estaba)
    db.execute(
        "INSERT INTO reunion_participantes (reunion_id, usuario_id) VALUES (%s, %s) "
        "ON DUPLICATE KEY UPDATE unido_en = NOW(), salido_en = NULL",
        (reunion["id"], usuario["id"])
    )

    return {"ok": True, "reunion": reunion, "codigo": codigo.upper()}


@router.get("/historial")
async def historial(usuario = Depends(get_usuario_actual)):
    """Retorna las reuniones en las que el usuario ha participado."""
    rows = db.fetchall(
        """SELECT r.codigo, r.nombre, r.creada_en, r.activa,
                  COUNT(m.id) AS total_mensajes
           FROM reuniones r
           JOIN reunion_participantes rp ON rp.reunion_id = r.id
           LEFT JOIN mensajes m ON m.reunion_id = r.id
           WHERE rp.usuario_id = %s
           GROUP BY r.id
           ORDER BY r.creada_en DESC
           LIMIT 50""",
        (usuario["id"],)
    )
    return {"reuniones": rows}


@router.get("/{codigo}/mensajes")
async def mensajes_reunion(codigo: str, usuario = Depends(get_usuario_actual)):
    """Retorna el historial de mensajes de una sala."""
    reunion = db.fetchone("SELECT id FROM reuniones WHERE codigo = %s", (codigo.upper(),))
    if not reunion:
        raise HTTPException(404, "Sala no encontrada")

    mensajes = db.fetchall(
        """SELECT m.id, m.texto_original, m.texto_corregido, m.tipo, m.confianza,
                  m.enviado_en, u.nombre AS autor, u.id AS autor_id
           FROM mensajes m
           JOIN usuarios u ON u.id = m.usuario_id
           WHERE m.reunion_id = %s
           ORDER BY m.enviado_en ASC
           LIMIT 200""",
        (reunion["id"],)
    )
    return {"mensajes": mensajes}


@router.post("/{codigo}/mensajes")
async def guardar_mensaje(codigo: str, body: dict, usuario = Depends(get_usuario_actual)):
    """Guarda un mensaje en la base de datos (llamado desde el WS handler)."""
    reunion = db.fetchone("SELECT id FROM reuniones WHERE codigo = %s", (codigo.upper(),))
    if not reunion:
        raise HTTPException(404, "Sala no encontrada")

    msg_id = db.insert(
        """INSERT INTO mensajes
           (reunion_id, usuario_id, texto_original, texto_corregido, tipo, confianza)
           VALUES (%s, %s, %s, %s, %s, %s)""",
        (
            reunion["id"],
            usuario["id"],
            body.get("texto_original", ""),
            body.get("texto_corregido"),
            body.get("tipo", "senas"),
            body.get("confianza"),
        )
    )
    return {"ok": True, "mensaje_id": msg_id}


@router.delete("/{codigo}/cerrar")
async def cerrar_reunion(codigo: str, usuario = Depends(get_usuario_actual)):
    """Cierra una sala (solo el creador o admin puede hacerlo)."""
    reunion = db.fetchone(
        "SELECT id, creador_id FROM reuniones WHERE codigo = %s", (codigo.upper(),)
    )
    if not reunion:
        raise HTTPException(404, "Sala no encontrada")
    if reunion["creador_id"] != usuario["id"] and usuario["rol"] != "admin":
        raise HTTPException(403, "No tienes permiso para cerrar esta sala")

    db.execute(
        "UPDATE reuniones SET activa = 0, cerrada_en = NOW() WHERE id = %s",
        (reunion["id"],)
    )
    return {"ok": True, "mensaje": "Sala cerrada"}

@router.get("/{codigo}/info")
async def info_reunion(codigo: str, usuario = Depends(get_usuario_actual)):
    """Info de la sala + si el usuario actual es el creador."""
    reunion = db.fetchone(
        "SELECT id, nombre, creador_id, activa FROM reuniones WHERE codigo = %s",
        (codigo.upper(),)
    )
    if not reunion:
        raise HTTPException(404, "Sala no encontrada")
    return {
        "reunion":    reunion,
        "es_creador": reunion["creador_id"] == usuario["id"]
    }