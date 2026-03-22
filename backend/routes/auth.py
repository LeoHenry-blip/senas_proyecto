"""
routes/auth.py
==============
Rutas de autenticación: registro, login, perfil, OAuth Google.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Optional

from core.auth import registrar_usuario, login_usuario, login_o_registrar_google, get_usuario_actual

router = APIRouter(prefix="/auth", tags=["Autenticación"])


class RegistroBody(BaseModel):
    nombre: str
    email: str
    password: str
    rol: Optional[str] = "usuario"


class LoginBody(BaseModel):
    email: str
    password: str


class GoogleLoginBody(BaseModel):
    google_id: str
    email: str
    nombre: str


@router.post("/registro", status_code=201)
async def registro(body: RegistroBody):
    """Registra un nuevo usuario con email y contraseña."""
    user_id = registrar_usuario(body.nombre, body.email, body.password, body.rol)
    return {"ok": True, "usuario_id": user_id, "mensaje": "Usuario creado exitosamente"}


@router.post("/login")
async def login(body: LoginBody):
    """Login con email y contraseña. Retorna JWT."""
    return login_usuario(body.email, body.password)


@router.post("/google")
async def login_google(body: GoogleLoginBody):
    """Login o registro mediante OAuth de Google."""
    return login_o_registrar_google(body.google_id, body.email, body.nombre)


@router.get("/perfil")
async def perfil(usuario=Depends(get_usuario_actual)):
    """Retorna los datos del usuario autenticado."""
    return {"usuario": usuario}
