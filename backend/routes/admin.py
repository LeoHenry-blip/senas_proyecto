"""
routes/admin.py
===============
Panel de administración: gestos, configuración, usuarios, estadísticas.
Todas las rutas requieren rol admin.
v3.1 — agrega endpoint /admin/test-dtw para depuración de palabras con movimiento
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
    rows = db.fetchall(
        "SELECT id, nombre, email, rol, activo, creado_en, ultimo_login FROM usuarios ORDER BY creado_en DESC"
    )
    return {"usuarios": rows}


@router.patch("/usuarios/{user_id}")
async def actualizar_usuario(user_id: int, body: dict, admin = Depends(requerir_admin)):
    campos_permitidos = {"rol": str, "activo": int}
    updates = []
    params  = []
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
    total_usuarios    = db.fetchone("SELECT COUNT(*) AS n FROM usuarios")["n"]
    total_reuniones   = db.fetchone("SELECT COUNT(*) AS n FROM reuniones")["n"]
    reuniones_activas = db.fetchone("SELECT COUNT(*) AS n FROM reuniones WHERE activa=1")["n"]
    total_mensajes    = db.fetchone("SELECT COUNT(*) AS n FROM mensajes")["n"]
    total_gestos      = db.fetchone("SELECT COUNT(*) AS n FROM gestos")["n"]
    gestos_por_tipo   = db.fetchall("SELECT tipo, COUNT(*) AS cantidad FROM gestos GROUP BY tipo")
    return {
        "usuarios": total_usuarios, "reuniones": total_reuniones,
        "reuniones_activas": reuniones_activas, "mensajes": total_mensajes,
        "gestos": total_gestos, "gestos_por_tipo": gestos_por_tipo,
    }


# =============================================================================
# GESTOS
# =============================================================================

class GuestoBody(BaseModel):
    nombre: str
    tipo: Optional[str] = "letter"
    descripcion: Optional[str] = ""
    landmarks: List[float]
    secuencia: Optional[List[List[List[float]]]] = None   # [reps][frames][63f]
    muestras_usadas: Optional[int] = 1


@router.get("/gestos")
async def listar_gestos(admin = Depends(requerir_admin)):
    rows = db.fetchall(
        """SELECT g.id, g.nombre, g.tipo, g.descripcion, g.muestras_usadas,
                  g.creado_en, g.actualizado_en,
                  u.nombre AS creado_por_nombre
           FROM gestos g
           LEFT JOIN usuarios u ON u.id = g.creado_por
           ORDER BY g.tipo, g.nombre"""
    )
    for row in rows:
        row_full = db.fetchone(
            "SELECT landmarks_json, secuencia_json FROM gestos WHERE id = %s", (row["id"],)
        )
        row["landmarks"] = json.loads(row_full["landmarks_json"])
        if row["tipo"] == "word" and row_full.get("secuencia_json"):
            row["secuencia"] = json.loads(row_full["secuencia_json"])
    return {"gestos": rows}


@router.post("/gestos", status_code=201)
async def crear_gesto(body: GuestoBody, admin = Depends(requerir_admin)):
    """
    Crea o actualiza un gesto.
    Letras  → landmarks normalizados.
    Palabras → landmarks (primer frame) + secuencia DTW completa [reps][frames][63f].
    """
    if len(body.landmarks) != 63:
        raise HTTPException(400, f"Se esperan 63 landmarks, se recibieron {len(body.landmarks)}")
    if body.tipo == "word" and not body.secuencia:
        raise HTTPException(400, "Las palabras con movimiento requieren el campo 'secuencia'")

    vector = np.array(body.landmarks, dtype=np.float32)
    norma  = np.linalg.norm(vector)
    if norma > 0:
        vector = vector / norma
    landmarks_json = json.dumps(vector.tolist())

    secuencia_json = None
    if body.secuencia:
        if sum(len(rep) for rep in body.secuencia) < 3:
            raise HTTPException(400, "La secuencia tiene muy pocos frames")
        for i, rep in enumerate(body.secuencia):
            for j, frame in enumerate(rep):
                if len(frame) != 63:
                    raise HTTPException(400, f"Frame {j} rep {i}: se esperan 63 floats")
        secuencia_json = json.dumps(body.secuencia)

    nombre_upper = body.nombre.upper().strip()
    existente    = db.fetchone("SELECT id FROM gestos WHERE nombre = %s", (nombre_upper,))

    if existente:
        db.execute(
            """UPDATE gestos SET tipo=%s, descripcion=%s, landmarks_json=%s,
               secuencia_json=%s, muestras_usadas=%s, creado_por=%s WHERE nombre=%s""",
            (body.tipo, body.descripcion, landmarks_json,
             secuencia_json, body.muestras_usadas, admin["id"], nombre_upper)
        )
        accion = "actualizado"
    else:
        db.insert(
            """INSERT INTO gestos (nombre, tipo, descripcion, landmarks_json,
               secuencia_json, muestras_usadas, creado_por) VALUES (%s,%s,%s,%s,%s,%s,%s)""",
            (nombre_upper, body.tipo, body.descripcion, landmarks_json,
             secuencia_json, body.muestras_usadas, admin["id"])
        )
        accion = "creado"

    exportar_gestos_a_json()
    _recargar_reconocedor()
    print(f"[Admin] {'Palabra DTW' if body.tipo == 'word' else 'Letra'} '{nombre_upper}' {accion}")
    return {"ok": True, "accion": accion, "nombre": nombre_upper}


@router.delete("/gestos/{nombre}")
async def eliminar_gesto(nombre: str, admin = Depends(requerir_admin)):
    nombre_upper = nombre.upper()
    if db.execute("DELETE FROM gestos WHERE nombre = %s", (nombre_upper,)) == 0:
        raise HTTPException(404, f"Gesto '{nombre_upper}' no encontrado")
    exportar_gestos_a_json()
    _recargar_reconocedor()
    return {"ok": True, "mensaje": f"Gesto '{nombre_upper}' eliminado"}


@router.post("/gestos/exportar-json")
async def exportar_json(admin = Depends(requerir_admin)):
    if not exportar_gestos_a_json():
        raise HTTPException(500, "Error al exportar gestos")
    total    = db.fetchone("SELECT COUNT(*) AS n FROM gestos")["n"]
    letras   = db.fetchone("SELECT COUNT(*) AS n FROM gestos WHERE tipo != 'word'")["n"]
    palabras = db.fetchone("SELECT COUNT(*) AS n FROM gestos WHERE tipo = 'word'")["n"]
    return {"ok": True, "total_exportados": total, "letras": letras, "palabras": palabras}


# =============================================================================
# TEST DTW — depuración de palabras con movimiento desde el panel de test
# =============================================================================

class TestDTWBody(BaseModel):
    # Lista de frames capturados en el navegador: [[63f] x N]
    # Cada frame ya viene normalizado (extraído por /api/admin/extraer-landmarks)
    frames: List[List[float]]


@router.post("/test-dtw")
async def test_dtw(body: TestDTWBody, admin = Depends(requerir_admin)):
    """
    Evalúa una secuencia de frames contra todas las palabras entrenadas con DTW.

    El panel de test acumula frames mientras el usuario hace el movimiento,
    luego los envía aquí. Se ejecuta el mismo DTW que usa el reconocedor
    en la sala de reunión, devolviendo el ranking completo para depuración.

    Returns:
        mejor:            La palabra reconocida si supera el umbral (o null)
        ranking:          Todas las palabras ordenadas por distancia DTW
        frames_recibidos: Cuántos frames llegaron
        palabras_en_bd:   Cuántas palabras están entrenadas
        umbral_dtw:       Umbral actual configurado
    """
    if len(body.frames) < 5:
        raise HTTPException(400, f"Mínimo 5 frames, se recibieron {len(body.frames)}")

    for i, frame in enumerate(body.frames):
        if len(frame) != 63:
            raise HTTPException(400, f"Frame {i}: se esperan 63 floats, hay {len(frame)}")

    bd = _obtener_bd()
    if bd is None:
        raise HTTPException(503, "Reconocedor no disponible. Reinicia el servidor.")

    if not bd.palabras:
        return {
            "mejor": None, "ranking": [],
            "frames_recibidos": len(body.frames),
            "palabras_en_bd": 0,
            "umbral_dtw": bd.umbral_dtw,
            "mensaje": "No hay palabras con movimiento entrenadas en la BD"
        }

    from base_datos import dtw_distancia_rapida

    secuencia = np.array(body.frames, dtype=np.float32)
    ranking   = []

    for nombre, datos in bd.palabras.items():
        mejor_dist = float('inf')
        for seq_ref in datos["secuencias"]:
            dist = dtw_distancia_rapida(secuencia, seq_ref, ventana=3)
            if dist < mejor_dist:
                mejor_dist = dist
        confianza = max(0.0, min(1.0, 1.0 - mejor_dist / bd.umbral_dtw))
        ranking.append({
            "nombre":    nombre,
            "confianza": round(confianza, 4),
            "distancia": round(mejor_dist, 4),
        })

    ranking.sort(key=lambda x: x["distancia"])

    mejor_reconocido = None
    if ranking and ranking[0]["distancia"] <= bd.umbral_dtw:
        mejor_reconocido = ranking[0]

    return {
        "mejor":            mejor_reconocido,
        "ranking":          ranking,
        "frames_recibidos": len(body.frames),
        "palabras_en_bd":   len(bd.palabras),
        "umbral_dtw":       bd.umbral_dtw,
    }


# =============================================================================
# CONFIGURACIÓN DEL RECONOCEDOR
# =============================================================================

@router.get("/config")
async def obtener_config(admin = Depends(requerir_admin)):
    rows = db.fetchall("SELECT clave, valor, descripcion FROM configuracion ORDER BY clave")
    return {"config": rows}


@router.patch("/config")
async def actualizar_config(body: dict, admin = Depends(requerir_admin)):
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
        _aplicar_config_reconocedor()
    return {"ok": True, "actualizadas": actualizadas}


# =============================================================================
# HELPERS INTERNOS
# =============================================================================

def _obtener_bd():
    """
    Retorna la BaseDatosGestos activa.
    Primero intenta el reconocedor en memoria; si falla, carga desde JSON.
    """
    try:
        from main import reconocedor_global
        if reconocedor_global and reconocedor_global.db:
            return reconocedor_global.db
    except Exception:
        pass
    try:
        from base_datos import BaseDatosGestos
        return BaseDatosGestos("gestos.json")
    except Exception as e:
        print(f"[Admin] No se pudo cargar BD para test DTW: {e}")
        return None


def _recargar_reconocedor():
    try:
        from main import reconocedor_global
        if reconocedor_global:
            reconocedor_global.db = __import__(
                'base_datos', fromlist=['BaseDatosGestos']
            ).BaseDatosGestos("gestos.json")
            print("[Admin] Reconocedor recargado")
    except Exception as e:
        print(f"[Admin] No se pudo recargar reconocedor: {e}")


def _aplicar_config_reconocedor():
    try:
        from main import reconocedor_global
        if not reconocedor_global:
            return
        reconocedor_global.db.umbral_estatico  = float(get_config("umbral_similitud", "0.35"))
        reconocedor_global.buffer_tamano        = int(get_config("frames_suavizado", "10"))
        reconocedor_global.tiempo_confirmacion  = float(get_config("tiempo_confirmacion", "1.2"))
        reconocedor_global.tiempo_pausa_letra   = float(get_config("tiempo_pausa_letra", "1.5"))
        reconocedor_global.tiempo_pausa_palabra = float(get_config("tiempo_pausa_palabra", "3.0"))
        print("[Admin] Configuración aplicada en tiempo real")
    except Exception as e:
        print(f"[Admin] Error aplicando config: {e}")