"""
routes/admin.py
===============
Panel de administración: gestos, configuración, usuarios, estadísticas.
Todas las rutas requieren rol admin.
"""

import json
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional, List

from core.auth import requerir_admin, get_usuario_actual
from db.database import db, get_config, set_config, exportar_gestos_a_json

router = APIRouter(prefix="/admin", tags=["Administración"])


# =============================================================================
# USUARIOS
# =============================================================================

@router.get("/usuarios")
async def listar_usuarios(admin = Depends(requerir_admin)):
    """Lista todos los usuarios del sistema."""
    rows = db.fetchall(
        "SELECT id, nombre, email, rol, activo, creado_en, ultimo_login FROM usuarios ORDER BY creado_en DESC"
    )
    return {"usuarios": rows}


@router.patch("/usuarios/{user_id}")
async def actualizar_usuario(user_id: int, body: dict, admin = Depends(requerir_admin)):
    """Cambia rol o estado activo de un usuario."""
    campos_permitidos = {"rol": str, "activo": int}
    updates = []
    params = []

    for campo, tipo in campos_permitidos.items():
        if campo in body:
            updates.append(f"{campo} = %s")
            params.append(tipo(body[campo]))

    if not updates:
        raise HTTPException(400, "Sin campos válidos para actualizar")

    params.append(user_id)
    db.execute(f"UPDATE usuarios SET {', '.join(updates)} WHERE id = %s", params)
    return {"ok": True}


# =============================================================================
# ESTADÍSTICAS
# =============================================================================

@router.get("/stats")
async def estadisticas(admin = Depends(requerir_admin)):
    """Estadísticas generales del sistema."""
    total_usuarios   = db.fetchone("SELECT COUNT(*) AS n FROM usuarios")["n"]
    total_reuniones  = db.fetchone("SELECT COUNT(*) AS n FROM reuniones")["n"]
    reuniones_activas = db.fetchone("SELECT COUNT(*) AS n FROM reuniones WHERE activa=1")["n"]
    total_mensajes   = db.fetchone("SELECT COUNT(*) AS n FROM mensajes")["n"]
    total_gestos     = db.fetchone("SELECT COUNT(*) AS n FROM gestos")["n"]

    gestos_por_tipo = db.fetchall(
        "SELECT tipo, COUNT(*) AS cantidad FROM gestos GROUP BY tipo"
    )

    return {
        "usuarios":        total_usuarios,
        "reuniones":       total_reuniones,
        "reuniones_activas": reuniones_activas,
        "mensajes":        total_mensajes,
        "gestos":          total_gestos,
        "gestos_por_tipo": gestos_por_tipo,
    }


# =============================================================================
# GESTOS  (entrenamiento desde el panel web)
# =============================================================================

class GuestoBody(BaseModel):
    nombre: str
    tipo: Optional[str] = "letter"
    descripcion: Optional[str] = ""
    landmarks: List[float]           # 63 floats (21 puntos × 3 coordenadas)
    muestras_usadas: Optional[int] = 1


@router.get("/gestos")
async def listar_gestos(admin = Depends(requerir_admin)):
    """Retorna todos los gestos registrados."""
    rows = db.fetchall(
        """SELECT g.id, g.nombre, g.tipo, g.descripcion, g.muestras_usadas,
                  g.creado_en, g.actualizado_en,
                  u.nombre AS creado_por_nombre
           FROM gestos g
           LEFT JOIN usuarios u ON u.id = g.creado_por
           ORDER BY g.tipo, g.nombre"""
    )
    # Incluir los landmarks como lista de floats
    for row in rows:
        row_full = db.fetchone("SELECT landmarks_json FROM gestos WHERE id = %s", (row["id"],))
        row["landmarks"] = json.loads(row_full["landmarks_json"])
    return {"gestos": rows}


@router.post("/gestos", status_code=201)
async def crear_gesto(body: GuestoBody, admin = Depends(requerir_admin)):
    """
    Crea o actualiza un gesto en la base de datos.
    Si ya existe el nombre, sobreescribe el vector de landmarks.
    """
    # Validar que sean exactamente 63 valores (21 puntos × xyz)
    if len(body.landmarks) != 63:
        raise HTTPException(400, f"Se esperan 63 valores de landmarks, se recibieron {len(body.landmarks)}")

    # Normalizar el vector (igual que en base_datos.py)
    vector = np.array(body.landmarks, dtype=np.float32)
    norma  = np.linalg.norm(vector)
    if norma > 0:
        vector = vector / norma
    landmarks_json = json.dumps(vector.tolist())

    nombre_upper = body.nombre.upper().strip()

    # Verificar si ya existe
    existente = db.fetchone("SELECT id FROM gestos WHERE nombre = %s", (nombre_upper,))

    if existente:
        db.execute(
            """UPDATE gestos
               SET tipo=%s, descripcion=%s, landmarks_json=%s,
                   muestras_usadas=%s, creado_por=%s
               WHERE nombre=%s""",
            (body.tipo, body.descripcion, landmarks_json,
             body.muestras_usadas, admin["id"], nombre_upper)
        )
        gesto_id = existente["id"]
        accion = "actualizado"
    else:
        gesto_id = db.insert(
            """INSERT INTO gestos
               (nombre, tipo, descripcion, landmarks_json, muestras_usadas, creado_por)
               VALUES (%s, %s, %s, %s, %s, %s)""",
            (nombre_upper, body.tipo, body.descripcion,
             landmarks_json, body.muestras_usadas, admin["id"])
        )
        accion = "creado"

    # Exportar a JSON para que el reconocedor local lo use
    exportar_gestos_a_json()

    # Recargar el reconocedor en memoria
    _recargar_reconocedor()

    return {"ok": True, "gesto_id": gesto_id, "accion": accion, "nombre": nombre_upper}


@router.delete("/gestos/{nombre}")
async def eliminar_gesto(nombre: str, admin = Depends(requerir_admin)):
    """Elimina un gesto por su nombre."""
    nombre_upper = nombre.upper()
    afectadas = db.execute("DELETE FROM gestos WHERE nombre = %s", (nombre_upper,))
    if afectadas == 0:
        raise HTTPException(404, f"Gesto '{nombre_upper}' no encontrado")

    exportar_gestos_a_json()
    _recargar_reconocedor()
    return {"ok": True, "mensaje": f"Gesto '{nombre_upper}' eliminado"}


@router.post("/gestos/exportar-json")
async def exportar_json(admin = Depends(requerir_admin)):
    """Exporta todos los gestos de MySQL al archivo gestos.json."""
    ok = exportar_gestos_a_json()
    if not ok:
        raise HTTPException(500, "Error al exportar gestos")
    total = db.fetchone("SELECT COUNT(*) AS n FROM gestos")["n"]
    return {"ok": True, "total_exportados": total}


# =============================================================================
# CONFIGURACIÓN DEL RECONOCEDOR
# =============================================================================

@router.get("/config")
async def obtener_config(admin = Depends(requerir_admin)):
    """Retorna toda la configuración actual del reconocedor."""
    rows = db.fetchall("SELECT clave, valor, descripcion FROM configuracion ORDER BY clave")
    return {"config": rows}


@router.patch("/config")
async def actualizar_config(body: dict, admin = Depends(requerir_admin)):
    """
    Actualiza uno o varios parámetros de configuración.
    Body: {"umbral_similitud": "0.30", "frames_suavizado": "8"}
    """
    claves_permitidas = {
        "umbral_similitud", "frames_suavizado", "confianza_minima",
        "tiempo_confirmacion", "tiempo_pausa_letra", "tiempo_pausa_palabra",
        "usar_ia_corrector", "velocidad_voz"
    }

    actualizadas = []
    for clave, valor in body.items():
        if clave not in claves_permitidas:
            continue
        set_config(clave, str(valor))
        actualizadas.append(clave)

    if actualizadas:
        # Aplicar cambios al reconocedor en memoria
        _aplicar_config_reconocedor()

    return {"ok": True, "actualizadas": actualizadas}


# =============================================================================
# HELPERS INTERNOS
# =============================================================================

def _recargar_reconocedor():
    """
    Recarga la base de datos del reconocedor después de agregar/eliminar gestos.
    Importa lazy para evitar dependencias circulares.
    """
    try:
        from main import reconocedor_global
        if reconocedor_global:
            reconocedor_global.db = __import__(
                'base_datos', fromlist=['BaseDatosGestos']
            ).BaseDatosGestos("gestos.json")
            print("[Admin] Reconocedor recargado con nuevos gestos")
    except Exception as e:
        print(f"[Admin] No se pudo recargar reconocedor: {e}")


def _aplicar_config_reconocedor():
    """Aplica los parámetros de configuración al reconocedor en tiempo real."""
    try:
        from main import reconocedor_global
        if not reconocedor_global:
            return

        from db.database import get_config

        umbral = float(get_config("umbral_similitud", "0.35"))
        reconocedor_global.db.umbral_similitud = umbral

        frames = int(get_config("frames_suavizado", "10"))
        reconocedor_global.buffer_tamano = frames

        t_conf = float(get_config("tiempo_confirmacion", "1.2"))
        reconocedor_global.tiempo_confirmacion = t_conf

        t_letra = float(get_config("tiempo_pausa_letra", "1.5"))
        reconocedor_global.tiempo_pausa_letra = t_letra

        t_palabra = float(get_config("tiempo_pausa_palabra", "3.0"))
        reconocedor_global.tiempo_pausa_palabra = t_palabra

        print("[Admin] Configuración del reconocedor actualizada en tiempo real")
    except Exception as e:
        print(f"[Admin] Error aplicando config: {e}")
