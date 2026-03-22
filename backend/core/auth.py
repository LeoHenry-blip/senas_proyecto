"""
core/auth.py
============
Autenticación JWT + bcrypt para login local.
También provee helpers para OAuth con Google.
"""

import bcrypt                          # Hash seguro de contraseñas
import jwt                             # JSON Web Tokens
import os
from datetime import datetime, timedelta
from typing import Optional, Dict
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from db.database import db

# Clave secreta para firmar los JWT (en producción: variable de entorno larga)
JWT_SECRET     = os.getenv("JWT_SECRET", "senas_v2_super_secret_change_in_production")
JWT_ALGORITHM  = "HS256"
JWT_EXPIRE_MIN = int(os.getenv("JWT_EXPIRE_MIN", "1440"))  # 24 horas por defecto

# Esquema de seguridad para FastAPI (lee el header Authorization: Bearer <token>)
bearer_scheme = HTTPBearer(auto_error=False)


# =============================================================================
# HASH DE CONTRASEÑAS
# =============================================================================

def hashear_password(password: str) -> str:
    """
    Genera el hash bcrypt de una contraseña.

    Args:
        password: Contraseña en texto plano

    Returns:
        Hash bcrypt como string
    """
    # bcrypt genera un salt aleatorio internamente (rounds=12 = buen balance)
    salt   = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verificar_password(password: str, hash_guardado: str) -> bool:
    """
    Verifica si una contraseña coincide con su hash almacenado.

    Args:
        password: Contraseña en texto plano proporcionada por el usuario
        hash_guardado: Hash bcrypt guardado en la base de datos

    Returns:
        True si la contraseña es correcta
    """
    return bcrypt.checkpw(
        password.encode("utf-8"),
        hash_guardado.encode("utf-8")
    )


# =============================================================================
# TOKENS JWT
# =============================================================================

def crear_token(usuario_id: int, email: str, rol: str) -> str:
    """
    Crea un token JWT con los datos del usuario.

    Args:
        usuario_id: ID del usuario en MySQL
        email: Email del usuario
        rol: 'usuario' o 'admin'

    Returns:
        Token JWT firmado como string
    """
    # Payload del token
    payload = {
        "sub":        str(usuario_id),   # Subject = ID del usuario
        "email":      email,
        "rol":        rol,
        "iat":        datetime.utcnow(),  # Issued at
        "exp":        datetime.utcnow() + timedelta(minutes=JWT_EXPIRE_MIN),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decodificar_token(token: str) -> Optional[Dict]:
    """
    Decodifica y valida un token JWT.

    Args:
        token: Token JWT

    Returns:
        Payload decodificado o None si es inválido/expirado
    """
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return None   # Token expirado
    except jwt.InvalidTokenError:
        return None   # Token inválido


# =============================================================================
# DEPENDENCIAS FASTAPI
# =============================================================================

async def get_usuario_actual(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme)
) -> Dict:
    """
    Dependencia de FastAPI: extrae y valida el usuario del token Bearer.
    Lanza 401 si el token es inválido o no está presente.

    Uso en rutas:
        @app.get("/ruta-protegida")
        async def ruta(usuario = Depends(get_usuario_actual)):
            ...
    """
    # Verificar que se envió el header Authorization
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token de autenticación requerido",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Decodificar el token
    payload = decodificar_token(credentials.credentials)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token inválido o expirado",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Verificar que el usuario existe y está activo en la BD
    usuario = db.fetchone(
        "SELECT id, nombre, email, rol, activo FROM usuarios WHERE id = %s",
        (int(payload["sub"]),)
    )

    if not usuario or not usuario["activo"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Usuario no encontrado o deshabilitado"
        )

    return usuario


async def requerir_admin(
    usuario: Dict = Depends(get_usuario_actual)
) -> Dict:
    """
    Dependencia FastAPI: igual que get_usuario_actual pero exige rol admin.
    Lanza 403 si el usuario no es admin.
    """
    if usuario["rol"] != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Se requieren permisos de administrador"
        )
    return usuario


# =============================================================================
# OPERACIONES DE USUARIO EN BD
# =============================================================================

def registrar_usuario(nombre: str, email: str, password: str, rol: str = "usuario") -> int:
    """
    Registra un nuevo usuario en MySQL.

    Args:
        nombre: Nombre completo
        email: Correo electrónico (debe ser único)
        password: Contraseña en texto plano
        rol: 'usuario' o 'admin'

    Returns:
        ID del usuario creado

    Raises:
        HTTPException 409 si el email ya existe
    """
    # Verificar que el email no esté registrado
    existente = db.fetchone("SELECT id FROM usuarios WHERE email = %s", (email,))
    if existente:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="El email ya está registrado"
        )

    # Hashear la contraseña antes de guardar
    password_hash = hashear_password(password)

    # Insertar el usuario
    user_id = db.insert(
        "INSERT INTO usuarios (nombre, email, password_hash, rol) VALUES (%s, %s, %s, %s)",
        (nombre, email, password_hash, rol)
    )

    return user_id


def login_usuario(email: str, password: str) -> Dict:
    """
    Verifica credenciales y retorna el token JWT.

    Args:
        email: Email del usuario
        password: Contraseña en texto plano

    Returns:
        Diccionario con token y datos del usuario

    Raises:
        HTTPException 401 si las credenciales son incorrectas
    """
    # Buscar el usuario por email
    usuario = db.fetchone(
        "SELECT id, nombre, email, password_hash, rol, activo FROM usuarios WHERE email = %s",
        (email,)
    )

    # Verificar que existe y la contraseña es correcta
    if not usuario or not verificar_password(password, usuario["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email o contraseña incorrectos"
        )

    # Verificar que la cuenta está activa
    if not usuario["activo"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cuenta deshabilitada. Contacta al administrador."
        )

    # Actualizar último login
    db.execute(
        "UPDATE usuarios SET ultimo_login = NOW() WHERE id = %s",
        (usuario["id"],)
    )

    # Crear y retornar el token JWT
    token = crear_token(usuario["id"], usuario["email"], usuario["rol"])

    return {
        "token":   token,
        "usuario": {
            "id":     usuario["id"],
            "nombre": usuario["nombre"],
            "email":  usuario["email"],
            "rol":    usuario["rol"],
        }
    }


def login_o_registrar_google(google_id: str, email: str, nombre: str) -> Dict:
    """
    Login con Google OAuth: busca usuario por google_id o lo crea.

    Args:
        google_id: ID único de Google
        email: Email de la cuenta Google
        nombre: Nombre del usuario en Google

    Returns:
        Diccionario con token y datos del usuario
    """
    # Buscar por google_id primero
    usuario = db.fetchone(
        "SELECT id, nombre, email, rol, activo FROM usuarios WHERE google_id = %s",
        (google_id,)
    )

    if not usuario:
        # Buscar por email (podría tener cuenta local con el mismo email)
        usuario = db.fetchone(
            "SELECT id, nombre, email, rol, activo FROM usuarios WHERE email = %s",
            (email,)
        )
        if usuario:
            # Vincular google_id a la cuenta existente
            db.execute(
                "UPDATE usuarios SET google_id = %s WHERE id = %s",
                (google_id, usuario["id"])
            )
        else:
            # Crear cuenta nueva con Google (sin password)
            user_id = db.insert(
                "INSERT INTO usuarios (nombre, email, password_hash, google_id) "
                "VALUES (%s, %s, %s, %s)",
                (nombre, email, "GOOGLE_AUTH_NO_PASSWORD", google_id)
            )
            usuario = db.fetchone(
                "SELECT id, nombre, email, rol, activo FROM usuarios WHERE id = %s",
                (user_id,)
            )

    # Actualizar último login
    db.execute("UPDATE usuarios SET ultimo_login = NOW() WHERE id = %s", (usuario["id"],))

    # Crear token
    token = crear_token(usuario["id"], usuario["email"], usuario["rol"])

    return {
        "token":   token,
        "usuario": {
            "id":     usuario["id"],
            "nombre": usuario["nombre"],
            "email":  usuario["email"],
            "rol":    usuario["rol"],
        }
    }
