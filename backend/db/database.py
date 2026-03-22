"""
db/database.py
==============
Conexión y gestión de la base de datos MySQL.
Usa PyMySQL como driver y un pool de conexiones simple.
Crea todas las tablas automáticamente si no existen.
"""

import pymysql                         # Driver MySQL puro Python
import pymysql.cursors                 # Cursores con diccionario
from contextlib import contextmanager  # Para el context manager de conexión
from typing import Optional, List, Dict, Any
import os                              # Variables de entorno
from datetime import datetime
import json                            # Para serializar vectores de gestos


# =============================================================================
# CONFIGURACIÓN DE CONEXIÓN  (leer desde variables de entorno)
# =============================================================================
DB_CONFIG = {
    "host":     os.getenv("DB_HOST",     "localhost"),
    "port":     int(os.getenv("DB_PORT", "3306")),
    "user":     os.getenv("DB_USER",     "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME",     "senas_v2"),
    "charset":  "utf8mb4",
    # Retornar filas como diccionarios (campo: valor)
    "cursorclass": pymysql.cursors.DictCursor,
    # Reconectar automáticamente si se cae la conexión
    "autocommit": True,
}


# =============================================================================
# POOL SIMPLE DE CONEXIONES
# =============================================================================

class DatabaseManager:
    """
    Gestor de conexión a MySQL.
    Provee un context manager para obtener cursores de forma segura.
    """

    def __init__(self):
        self._connection: Optional[pymysql.Connection] = None

    def _get_connection(self) -> pymysql.Connection:
        """
        Retorna la conexión activa o crea una nueva si se cerró.
        """
        # Verificar si la conexión existe y está activa
        if self._connection is None or not self._connection.open:
            self._connection = pymysql.connect(**DB_CONFIG)
        try:
            # ping() reconecta automáticamente si se perdió la conexión
            self._connection.ping(reconnect=True)
        except Exception:
            self._connection = pymysql.connect(**DB_CONFIG)
        return self._connection

    @contextmanager
    def cursor(self):
        """
        Context manager que provee un cursor y maneja commit/rollback.

        Uso:
            with db.cursor() as cur:
                cur.execute("SELECT ...")
                rows = cur.fetchall()
        """
        conn = self._get_connection()
        cur = conn.cursor()
        try:
            yield cur
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            cur.close()

    def execute(self, sql: str, params=None) -> int:
        """
        Ejecuta un comando SQL y retorna el número de filas afectadas.

        Args:
            sql: Sentencia SQL con placeholders %s
            params: Tupla o lista de parámetros

        Returns:
            Número de filas afectadas
        """
        with self.cursor() as cur:
            cur.execute(sql, params or ())
            return cur.rowcount

    def fetchone(self, sql: str, params=None) -> Optional[Dict]:
        """Ejecuta SELECT y retorna la primera fila como diccionario."""
        with self.cursor() as cur:
            cur.execute(sql, params or ())
            return cur.fetchone()

    def fetchall(self, sql: str, params=None) -> List[Dict]:
        """Ejecuta SELECT y retorna todas las filas como lista de diccionarios."""
        with self.cursor() as cur:
            cur.execute(sql, params or ())
            return cur.fetchall()

    def insert(self, sql: str, params=None) -> int:
        """
        Ejecuta INSERT y retorna el ID del registro insertado (lastrowid).
        """
        with self.cursor() as cur:
            cur.execute(sql, params or ())
            return cur.lastrowid

    def close(self):
        """Cierra la conexión al finalizar."""
        if self._connection and self._connection.open:
            self._connection.close()


# Instancia global del gestor de base de datos
db = DatabaseManager()


# =============================================================================
# INICIALIZACIÓN: CREAR BASE DE DATOS Y TABLAS
# =============================================================================

def crear_base_datos_si_no_existe():
    """
    Crea la base de datos MySQL si no existe.
    Se conecta sin especificar database para poder crearla.
    """
    config_sin_db = {k: v for k, v in DB_CONFIG.items()
                     if k not in ("database", "cursorclass")}
    config_sin_db["cursorclass"] = pymysql.cursors.DictCursor

    conn = pymysql.connect(**config_sin_db)
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"CREATE DATABASE IF NOT EXISTS `{DB_CONFIG['database']}` "
                f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
            conn.commit()
        print(f"[DB] Base de datos '{DB_CONFIG['database']}' lista")
    finally:
        conn.close()


SQL_CREAR_TABLAS = """
-- ============================================================
-- TABLA: usuarios
-- Almacena los usuarios registrados en el sistema
-- ============================================================
CREATE TABLE IF NOT EXISTS usuarios (
    id            INT AUTO_INCREMENT PRIMARY KEY,
    nombre        VARCHAR(100)        NOT NULL,
    email         VARCHAR(150)        NOT NULL UNIQUE,
    password_hash VARCHAR(256)        NOT NULL,          -- bcrypt hash
    rol           ENUM('usuario','admin') DEFAULT 'usuario',
    avatar_url    VARCHAR(500)        DEFAULT NULL,      -- URL del avatar
    activo        TINYINT(1)          DEFAULT 1,         -- 0 = baneado
    google_id     VARCHAR(150)        DEFAULT NULL,      -- OAuth Google
    creado_en     DATETIME            DEFAULT CURRENT_TIMESTAMP,
    ultimo_login  DATETIME            DEFAULT NULL,
    INDEX idx_email (email),
    INDEX idx_google_id (google_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ============================================================
-- TABLA: reuniones
-- Cada sala de videollamada / chat
-- ============================================================
CREATE TABLE IF NOT EXISTS reuniones (
    id            INT AUTO_INCREMENT PRIMARY KEY,
    codigo        VARCHAR(12)         NOT NULL UNIQUE,   -- ej: ABC-123-XYZ
    nombre        VARCHAR(200)        DEFAULT NULL,
    creador_id    INT                 NOT NULL,
    activa        TINYINT(1)          DEFAULT 1,
    creada_en     DATETIME            DEFAULT CURRENT_TIMESTAMP,
    cerrada_en    DATETIME            DEFAULT NULL,
    FOREIGN KEY (creador_id) REFERENCES usuarios(id) ON DELETE CASCADE,
    INDEX idx_codigo (codigo),
    INDEX idx_activa (activa)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ============================================================
-- TABLA: reunion_participantes
-- Quién está (o estuvo) en cada reunión
-- ============================================================
CREATE TABLE IF NOT EXISTS reunion_participantes (
    id            INT AUTO_INCREMENT PRIMARY KEY,
    reunion_id    INT                 NOT NULL,
    usuario_id    INT                 NOT NULL,
    unido_en      DATETIME            DEFAULT CURRENT_TIMESTAMP,
    salido_en     DATETIME            DEFAULT NULL,
    FOREIGN KEY (reunion_id) REFERENCES reuniones(id) ON DELETE CASCADE,
    FOREIGN KEY (usuario_id) REFERENCES usuarios(id)  ON DELETE CASCADE,
    UNIQUE KEY uq_reunion_usuario (reunion_id, usuario_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ============================================================
-- TABLA: mensajes
-- Historial de mensajes de cada reunión
-- ============================================================
CREATE TABLE IF NOT EXISTS mensajes (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    reunion_id      INT                 NOT NULL,
    usuario_id      INT                 NOT NULL,
    texto_original  TEXT                NOT NULL,        -- Lo que detectó el sistema
    texto_corregido TEXT                DEFAULT NULL,    -- Después de corrección IA
    tipo            ENUM('senas','texto','sistema') DEFAULT 'senas',
    confianza       FLOAT               DEFAULT NULL,   -- 0-1
    enviado_en      DATETIME            DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (reunion_id) REFERENCES reuniones(id) ON DELETE CASCADE,
    FOREIGN KEY (usuario_id) REFERENCES usuarios(id),
    INDEX idx_reunion_tiempo (reunion_id, enviado_en)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ============================================================
-- TABLA: gestos
-- Base de datos de gestos del reconocedor (sincronizada con gestos.json)
-- ============================================================
CREATE TABLE IF NOT EXISTS gestos (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    nombre          VARCHAR(50)         NOT NULL UNIQUE,  -- "A", "HOLA", etc.
    tipo            ENUM('letter','word','custom') DEFAULT 'letter',
    descripcion     VARCHAR(300)        DEFAULT NULL,
    landmarks_json  MEDIUMTEXT          NOT NULL,        -- Array JSON de 63 floats
    muestras_usadas INT                 DEFAULT 1,
    creado_por      INT                 DEFAULT NULL,    -- usuario admin que lo entrenó
    creado_en       DATETIME            DEFAULT CURRENT_TIMESTAMP,
    actualizado_en  DATETIME            DEFAULT CURRENT_TIMESTAMP
                                        ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (creado_por) REFERENCES usuarios(id) ON DELETE SET NULL,
    INDEX idx_nombre (nombre),
    INDEX idx_tipo (tipo)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ============================================================
-- TABLA: configuracion
-- Parámetros del reconocedor ajustables desde el panel admin
-- ============================================================
CREATE TABLE IF NOT EXISTS configuracion (
    clave   VARCHAR(100) PRIMARY KEY,
    valor   VARCHAR(500) NOT NULL,
    descripcion VARCHAR(300) DEFAULT NULL,
    actualizado_en DATETIME DEFAULT CURRENT_TIMESTAMP
                             ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ============================================================
-- TABLA: sesiones
-- Tokens JWT activos (para invalidación manual)
-- ============================================================
CREATE TABLE IF NOT EXISTS sesiones (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    usuario_id  INT          NOT NULL,
    token_hash  VARCHAR(256) NOT NULL,
    expira_en   DATETIME     NOT NULL,
    creada_en   DATETIME     DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (usuario_id) REFERENCES usuarios(id) ON DELETE CASCADE,
    INDEX idx_token (token_hash),
    INDEX idx_expira (expira_en)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

# Configuración por defecto del reconocedor
CONFIG_DEFAULT = [
    ("umbral_similitud",     "0.35",  "Distancia euclidiana máxima para reconocer un gesto"),
    ("frames_suavizado",     "10",    "Cantidad de frames para el buffer de suavizado"),
    ("confianza_minima",     "0.40",  "Confianza mínima para aceptar un gesto (0-1)"),
    ("tiempo_confirmacion",  "1.2",   "Segundos que debe sostenerse un gesto para confirmarse"),
    ("tiempo_pausa_letra",   "1.5",   "Segundos de pausa para separar letras"),
    ("tiempo_pausa_palabra", "3.0",   "Segundos de pausa para separar palabras"),
    ("usar_ia_corrector",    "true",  "Activar corrección con IA externa (requiere API key)"),
    ("velocidad_voz",        "150",   "Velocidad de síntesis de voz (WPM)"),
]


def inicializar_db():
    """
    Punto de entrada principal: crea la DB, las tablas y los datos por defecto.
    Llamar una vez al arrancar el servidor.
    """
    print("[DB] Inicializando base de datos MySQL...")

    # 1. Crear la base de datos si no existe
    crear_base_datos_si_no_existe()

    # 2. Crear todas las tablas
    # Dividir el bloque SQL en sentencias individuales y ejecutar cada una
    sentencias = [s.strip() for s in SQL_CREAR_TABLAS.split(";") if s.strip()]
    for sql in sentencias:
        if sql:
            try:
                db.execute(sql)
            except Exception as e:
                # Ignorar errores de "ya existe" (IF NOT EXISTS los previene)
                print(f"[DB] Aviso al crear tabla: {e}")

    print("[DB] Tablas creadas/verificadas")

    # 3. Insertar configuración por defecto si no existe
    for clave, valor, desc in CONFIG_DEFAULT:
        db.execute(
            "INSERT IGNORE INTO configuracion (clave, valor, descripcion) VALUES (%s, %s, %s)",
            (clave, valor, desc)
        )

    print("[DB] Configuración por defecto lista")
    print("[DB] ✅ Base de datos lista para usar")


# =============================================================================
# HELPERS REUTILIZABLES
# =============================================================================

def get_config(clave: str, default: str = "") -> str:
    """
    Obtiene un valor de configuración desde la tabla configuracion.

    Args:
        clave: Nombre de la configuración
        default: Valor por defecto si no existe

    Returns:
        Valor como string
    """
    row = db.fetchone("SELECT valor FROM configuracion WHERE clave = %s", (clave,))
    return row["valor"] if row else default


def set_config(clave: str, valor: str) -> None:
    """Actualiza o inserta un valor de configuración."""
    db.execute(
        "INSERT INTO configuracion (clave, valor) VALUES (%s, %s) "
        "ON DUPLICATE KEY UPDATE valor = %s, actualizado_en = NOW()",
        (clave, valor, valor)
    )


def sincronizar_gestos_desde_json(ruta_json: str = "gestos.json") -> int:
    """
    Importa gestos desde el archivo gestos.json a la tabla MySQL.
    No sobreescribe gestos ya existentes en MySQL.

    Args:
        ruta_json: Ruta al archivo JSON de gestos

    Returns:
        Número de gestos importados
    """
    import os
    if not os.path.exists(ruta_json):
        print(f"[DB] {ruta_json} no encontrado, saltando sincronización")
        return 0

    with open(ruta_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    gestures = data.get("gestures", {})
    importados = 0

    for nombre, info in gestures.items():
        landmarks_json = json.dumps(info.get("landmarks", []))
        try:
            db.execute(
                """INSERT IGNORE INTO gestos
                   (nombre, tipo, descripcion, landmarks_json, muestras_usadas)
                   VALUES (%s, %s, %s, %s, %s)""",
                (
                    nombre,
                    info.get("type", "letter"),
                    info.get("description", ""),
                    landmarks_json,
                    info.get("muestras_usadas", 1)
                )
            )
            importados += 1
        except Exception as e:
            print(f"[DB] Error importando gesto '{nombre}': {e}")

    print(f"[DB] Gestos sincronizados desde JSON: {importados}")
    return importados


def exportar_gestos_a_json(ruta_json: str = "gestos.json") -> bool:
    """
    Exporta todos los gestos de MySQL al archivo gestos.json.
    Útil para hacer backup o para que el reconocedor local los use.
    """
    rows = db.fetchall("SELECT * FROM gestos ORDER BY nombre")

    gestures = {}
    for row in rows:
        landmarks = json.loads(row["landmarks_json"])
        gestures[row["nombre"]] = {
            "name":           row["nombre"],
            "type":           row["tipo"],
            "description":    row["descripcion"] or "",
            "landmarks":      landmarks,
            "muestras_usadas": row["muestras_usadas"],
        }

    data = {
        "version":     "2.0",
        "description": "Base de datos exportada desde MySQL - Señas V2",
        "gestures":    gestures,
        "metadata": {
            "total_gestures": len(gestures),
            "exported_at":    datetime.now().isoformat()
        }
    }

    try:
        with open(ruta_json, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[DB] Gestos exportados a {ruta_json}: {len(gestures)}")
        return True
    except Exception as e:
        print(f"[DB] Error exportando: {e}")
        return False