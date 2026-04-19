"""
routes/admin.py  v4.1
=====================
Panel de administración con soporte SVM para gestos con movimiento.
Cambios v4.1:
  - guardar_muestra_svm: acepta 63 o 126 floats por frame (soporte 2 manos)
  - guardar_muestra_svm: mínimo de frames reducido de 5 a 3
  - entrenar_svm: acepta frames de 63 o 126 dims al filtrar muestras válidas
"""

import json
import os
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional, List

from core.auth import requerir_admin, get_usuario_actual
from db.database import db, get_config, set_config, exportar_gestos_a_json

router = APIRouter(prefix="/admin", tags=["Administración"])

RUTA_MUESTRAS = os.path.join(os.path.dirname(__file__), "..", "muestras_svm.json")


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
# GESTOS (sistema original — letras estáticas)
# =============================================================================

class GuestoBody(BaseModel):
    nombre: str
    tipo: Optional[str] = "letter"
    descripcion: Optional[str] = ""
    landmarks: List[float]
    secuencia: Optional[List[List[List[float]]]] = None
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
    if len(body.landmarks) != 63:
        raise HTTPException(400, f"Se esperan 63 landmarks, se recibieron {len(body.landmarks)}")

    vector = np.array(body.landmarks, dtype=np.float32)
    norma  = np.linalg.norm(vector)
    if norma > 0:
        vector = vector / norma
    landmarks_json = json.dumps(vector.tolist())

    secuencia_json = None
    if body.secuencia:
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
# SVM — MUESTRAS
# =============================================================================

class MuestraSVMBody(BaseModel):
    nombre:  str
    frames:  List[List[float]]
    agregar: Optional[bool] = True


@router.post("/muestras-svm", status_code=201)
async def guardar_muestra_svm(body: MuestraSVMBody, admin = Depends(requerir_admin)):
    """
    Guarda una muestra de entrenamiento SVM.
    Acepta frames de 63 floats (1 mano) o 126 floats (2 manos).
    Mínimo 3 frames por muestra.
    """
    # ── CORRECCIÓN 1: mínimo reducido de 5 a 3
    if len(body.frames) < 3:
        raise HTTPException(400, f"Mínimo 3 frames por muestra, se recibieron {len(body.frames)}")

    # ── CORRECCIÓN 2: acepta 63 o 126 floats por frame
    for i, frame in enumerate(body.frames):
        if len(frame) not in (63, 126):
            raise HTTPException(400, f"Frame {i}: se esperan 63 o 126 floats, hay {len(frame)}")

    nombre = body.nombre.upper().strip()

    muestras = _cargar_muestras_svm()

    if not body.agregar or nombre not in muestras:
        if not body.agregar:
            muestras[nombre] = []
        else:
            muestras.setdefault(nombre, [])

    muestras[nombre].append(body.frames)

    _guardar_muestras_svm(muestras)

    total = len(muestras[nombre])
    print(f"[Admin SVM] Muestra guardada para '{nombre}': {total} total")

    return {
        "ok":             True,
        "nombre":         nombre,
        "total_muestras": total,
        "gestos_en_bd":   list(muestras.keys()),
    }


@router.delete("/muestras-svm/{nombre}")
async def eliminar_muestras_gesto(nombre: str, admin = Depends(requerir_admin)):
    muestras = _cargar_muestras_svm()
    nombre_u = nombre.upper()
    if nombre_u not in muestras:
        raise HTTPException(404, f"No hay muestras para '{nombre_u}'")
    del muestras[nombre_u]
    _guardar_muestras_svm(muestras)
    return {"ok": True, "mensaje": f"Muestras de '{nombre_u}' eliminadas"}


@router.get("/muestras-svm")
async def listar_muestras_svm(admin = Depends(requerir_admin)):
    muestras = _cargar_muestras_svm()
    resumen  = {nombre: len(seqs) for nombre, seqs in muestras.items()}
    return {
        "gestos":          resumen,
        "total_gestos":    len(resumen),
        "total_muestras":  sum(resumen.values()),
        "listo_para_entrenar": len(resumen) >= 2 and all(n >= 5 for n in resumen.values()),
    }


# =============================================================================
# SVM — ENTRENAMIENTO
# =============================================================================

@router.post("/entrenar-svm")
async def entrenar_svm(admin = Depends(requerir_admin)):
    """
    Entrena el modelo SVM con todas las muestras acumuladas.
    Acepta muestras de 63 o 126 dims por frame.
    """
    muestras_raw = _cargar_muestras_svm()

    if len(muestras_raw) < 2:
        raise HTTPException(400,
            f"Necesitas al menos 2 gestos para entrenar. "
            f"Solo hay {len(muestras_raw)}: {list(muestras_raw.keys())}")

    muestras_np = {}
    for nombre, seqs in muestras_raw.items():
        arrs = []
        for seq in seqs:
            arr = np.array(seq, dtype=np.float32)
            # ── CORRECCIÓN 3: acepta 63 o 126 dims, mínimo 3 frames
            if arr.shape[1] in (63, 126) and len(arr) >= 3:
                arrs.append(arr)
        if arrs:
            muestras_np[nombre] = arrs

    if len(muestras_np) < 2:
        raise HTTPException(400, "No hay suficientes muestras válidas para entrenar")

    from modelo_svm import obtener_modelo, recargar_modelo, ModeloSVM, RUTA_MODELO
    modelo_tmp = ModeloSVM()
    resultado  = modelo_tmp.entrenar(muestras_np)

    if not resultado["ok"]:
        raise HTTPException(400, resultado.get("error", "Error entrenando"))

    modelo_tmp.guardar(RUTA_MODELO)
    recargar_modelo()

    print(f"[Admin SVM] Modelo entrenado: {resultado}")
    return resultado


# =============================================================================
# SVM — PREDICCIÓN (para panel de test)
# =============================================================================

class PredecirSVMBody(BaseModel):
    frames: List[List[float]]


@router.post("/predecir-svm")
async def predecir_svm(body: PredecirSVMBody, admin = Depends(requerir_admin)):
    if len(body.frames) < 3:
        raise HTTPException(400, f"Mínimo 3 frames, se recibieron {len(body.frames)}")

    from modelo_svm import obtener_modelo
    modelo = obtener_modelo()

    if not modelo.entrenado:
        return {
            "mejor":   None,
            "ranking": [],
            "error":   "Modelo SVM no entrenado. Ve al entrenador y entrena primero.",
        }

    secuencia = np.array(body.frames, dtype=np.float32)
    nombre, confianza = modelo.predecir(secuencia)
    ranking           = modelo.predecir_ranking(secuencia)

    return {
        "mejor":            {"nombre": nombre, "confianza": round(confianza, 4)} if nombre else None,
        "ranking":          ranking,
        "frames_recibidos": len(body.frames),
        "gestos_en_modelo": modelo.clases,
    }


# =============================================================================
# SVM — ESTADO
# =============================================================================

@router.get("/estado-svm")
async def estado_svm(admin = Depends(requerir_admin)):
    from modelo_svm import obtener_modelo, RUTA_MODELO
    modelo   = obtener_modelo()
    muestras = _cargar_muestras_svm()
    return {
        "modelo_entrenado":   modelo.entrenado,
        "gestos_en_modelo":   modelo.clases if modelo.entrenado else [],
        "modelo_existe":      os.path.exists(RUTA_MODELO),
        "muestras_guardadas": {n: len(s) for n, s in muestras.items()},
    }


# =============================================================================
# TEST DTW (sistema original — mantener compatibilidad)
# =============================================================================

class TestDTWBody(BaseModel):
    frames: List[List[float]]


@router.post("/test-dtw")
async def test_dtw(body: TestDTWBody, admin = Depends(requerir_admin)):
    if len(body.frames) < 5:
        raise HTTPException(400, f"Mínimo 5 frames, se recibieron {len(body.frames)}")

    for i, frame in enumerate(body.frames):
        if len(frame) != 63:
            raise HTTPException(400, f"Frame {i}: se esperan 63 floats, hay {len(frame)}")

    bd = _obtener_bd()
    if bd is None:
        raise HTTPException(503, "Reconocedor no disponible.")

    if not bd.palabras:
        return {
            "mejor": None, "ranking": [],
            "frames_recibidos": len(body.frames),
            "palabras_en_bd": 0,
            "umbral_dtw": bd.umbral_dtw,
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
        ranking.append({"nombre": nombre, "confianza": round(confianza, 4), "distancia": round(mejor_dist, 4)})

    ranking.sort(key=lambda x: x["distancia"])
    mejor = ranking[0] if ranking and ranking[0]["distancia"] <= bd.umbral_dtw else None

    return {
        "mejor": mejor, "ranking": ranking,
        "frames_recibidos": len(body.frames),
        "palabras_en_bd": len(bd.palabras),
        "umbral_dtw": bd.umbral_dtw,
    }


# =============================================================================
# CONFIGURACIÓN
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
# HELPERS
# =============================================================================

def _cargar_muestras_svm() -> dict:
    if not os.path.exists(RUTA_MUESTRAS):
        return {}
    try:
        with open(RUTA_MUESTRAS, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def _guardar_muestras_svm(muestras: dict) -> None:
    with open(RUTA_MUESTRAS, 'w', encoding='utf-8') as f:
        json.dump(muestras, f)


def _obtener_bd():
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
        print(f"[Admin] No se pudo cargar BD: {e}")
        return None


def _recargar_reconocedor():
    try:
        from main import reconocedor_global
        if reconocedor_global:
            reconocedor_global.db = __import__(
                'base_datos', fromlist=['BaseDatosGestos']
            ).BaseDatosGestos("gestos.json")
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
    except Exception as e:
        print(f"[Admin] Error aplicando config: {e}")