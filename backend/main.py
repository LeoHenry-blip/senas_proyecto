"""
main.py  (backend)
==================
Servidor FastAPI principal del sistema de Señas V2.
Monta todas las rutas, WebSocket y sirve el frontend estático.

Arrancar con:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi import WebSocket, Query

# ---- Módulos propios ----
from db.database import inicializar_db, sincronizar_gestos_desde_json
from routes.auth      import router as auth_router
from routes.reuniones import router as reuniones_router
from routes.admin     import router as admin_router
from routes.landmarks import router as landmarks_router
from websocket.endpoint import websocket_sala
from reconocedor import Reconocedor

# =============================================================================
# VARIABLE GLOBAL DEL RECONOCEDOR
# Accesible desde routes/admin.py para recargarlo en tiempo real
# =============================================================================
reconocedor_global: Reconocedor = None


# =============================================================================
# CREAR LA APLICACIÓN FASTAPI
# =============================================================================

app = FastAPI(
    title       = "Señas V2 - Sistema de Lenguaje de Señas",
    description = "API REST + WebSocket para traducción de lenguaje de señas en tiempo real",
    version     = "2.0.0",
    docs_url    = "/api/docs",     # Swagger UI
    redoc_url   = "/api/redoc",    # ReDoc
)


# =============================================================================
# CORS  (permitir peticiones desde el frontend en desarrollo)
# =============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # En producción: lista de dominios permitidos
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# =============================================================================
# EVENTOS DE INICIO Y CIERRE
# =============================================================================

@app.on_event("startup")
async def startup():
    """
    Al arrancar el servidor:
    1. Inicializar la base de datos MySQL
    2. Sincronizar gestos desde el JSON existente
    3. Cargar el reconocedor global
    """
    global reconocedor_global
    print("[App] Iniciando servidor Señas V2...")

    # Inicializar MySQL (crear tablas si no existen)
    inicializar_db()

    # Importar gestos desde el JSON del sistema V1
    sincronizar_gestos_desde_json("gestos.json")

    # Cargar reconocedor en memoria
    reconocedor_global = Reconocedor(ruta_gestos="gestos.json")

    print("[App] ✅ Servidor listo")


@app.on_event("shutdown")
async def shutdown():
    """Al cerrar el servidor, liberar recursos."""
    if reconocedor_global:
        pass  # Reconocedor no tiene recursos que liberar explícitamente
    print("[App] Servidor detenido")


# =============================================================================
# RUTAS API REST
# =============================================================================

app.include_router(auth_router,      prefix="/api")
app.include_router(reuniones_router, prefix="/api")
app.include_router(admin_router,     prefix="/api")
app.include_router(landmarks_router, prefix="/api")


# =============================================================================
# WEBSOCKET
# =============================================================================

@app.websocket("/ws/{sala_id}")
async def ws_endpoint(websocket: WebSocket, sala_id: str, token: str = Query(...)):
    """Endpoint WebSocket para una sala de reunión."""
    await websocket_sala(websocket, sala_id, token)


# =============================================================================
# ESTADO DEL SERVIDOR
# =============================================================================

@app.get("/api/health")
async def health():
    """Endpoint de health check para balanceadores de carga."""
    from websocket.manager import ws_manager
    return {
        "status":      "ok",
        "version":     "2.0.0",
        "websockets":  ws_manager.stats(),
    }


# =============================================================================
# SERVIR FRONTEND ESTÁTICO
# =============================================================================

# Servir archivos estáticos del frontend
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")

if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

    @app.get("/")
    async def root():
        return FileResponse(os.path.join(frontend_path, "index.html"))

    @app.get("/{path:path}")
    async def catch_all(path: str):
        """SPA: redirigir rutas desconocidas al index.html."""
        file_path = os.path.join(frontend_path, path)
        if os.path.exists(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(frontend_path, "index.html"))
