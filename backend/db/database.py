"""
db/database.py
==============
Conexión y gestión de la base de datos MySQL.
Usa PyMySQL como driver y un pool de conexiones simple.
Crea todas las tablas automáticamente si no existen.
v3.0 — soporte para secuencias DTW (palabras con movimiento)
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
        if self._connection is None or not self._connection.open:
            self._connection = pymysql.connect(**DB_CONFIG)
        try:
            self._connection.ping(reconnect=True)
        except Exception:
            self._connection = pymysql.connect(**DB_CONFIG)
        return self._connection

    @contextmanager
    def cursor(self):
        """
        Context manager que provee un cursor y maneja commit/rollback.
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
        """Ejecuta un comando SQL y retorna el número de filas afectadas."""
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
        """Ejecuta INSERT y retorna el ID del registro insertado (lastrowid)."""
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
-- ============================================================
CREATE TABLE IF NOT EXISTS usuarios (
    id            INT AUTO_INCREMENT PRIMARY KEY,
    nombre        VARCHAR(100)        NOT NULL,
    email         VARCHAR(150)        NOT NULL UNIQUE,
    password_hash VARCHAR(256)        NOT NULL,
    rol           ENUM('usuario','admin') DEFAULT 'usuario',
    avatar_url    VARCHAR(500)        DEFAULT NULL,
    activo        TINYINT(1)          DEFAULT 1,
    google_id     VARCHAR(150)        DEFAULT NULL,
    creado_en     DATETIME            DEFAULT CURRENT_TIMESTAMP,
    ultimo_login  DATETIME            DEFAULT NULL,
    INDEX idx_email (email),
    INDEX idx_google_id (google_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ============================================================
-- TABLA: reuniones
-- ============================================================
CREATE TABLE IF NOT EXISTS reuniones (
    id            INT AUTO_INCREMENT PRIMARY KEY,
    codigo        VARCHAR(12)         NOT NULL UNIQUE,
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
-- ============================================================
CREATE TABLE IF NOT EXISTS mensajes (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    reunion_id      INT                 NOT NULL,
    usuario_id      INT                 NOT NULL,
    texto_original  TEXT                NOT NULL,
    texto_corregido TEXT                DEFAULT NULL,
    tipo            ENUM('senas','texto','sistema') DEFAULT 'senas',
    confianza       FLOAT               DEFAULT NULL,
    enviado_en      DATETIME            DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (reunion_id) REFERENCES reuniones(id) ON DELETE CASCADE,
    FOREIGN KEY (usuario_id) REFERENCES usuarios(id),
    INDEX idx_reunion_tiempo (reunion_id, enviado_en)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ============================================================
-- TABLA: gestos
-- v3.0: agrega secuencia_json para palabras con movimiento (DTW)
-- ============================================================
CREATE TABLE IF NOT EXISTS gestos (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    nombre          VARCHAR(50)         NOT NULL UNIQUE,
    tipo            ENUM('letter','word','custom') DEFAULT 'letter',
    descripcion     VARCHAR(300)        DEFAULT NULL,
    landmarks_json  MEDIUMTEXT          NOT NULL,
    secuencia_json  LONGTEXT            DEFAULT NULL,
    muestras_usadas INT                 DEFAULT 1,
    creado_por      INT                 DEFAULT NULL,
    creado_en       DATETIME            DEFAULT CURRENT_TIMESTAMP,
    actualizado_en  DATETIME            DEFAULT CURRENT_TIMESTAMP
                                        ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (creado_por) REFERENCES usuarios(id) ON DELETE SET NULL,
    INDEX idx_nombre (nombre),
    INDEX idx_tipo (tipo)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ============================================================
-- TABLA: configuracion
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


def migrar_agregar_secuencia_json():
    """
    Migración segura: agrega columna secuencia_json si no existe.
    Se llama automáticamente en inicializar_db().
    Es idempotente — si ya existe la columna, no hace nada.
    """
    try:
        db.execute(
            "ALTER TABLE gestos ADD COLUMN secuencia_json LONGTEXT DEFAULT NULL"
        )
        print("[DB] Migración: columna secuencia_json agregada")
    except Exception:
        # MySQL lanza error si la columna ya existe — es esperado, ignorar
        pass


def inicializar_db():
    """
    Punto de entrada principal: crea la DB, las tablas y los datos por defecto.
    Llamar una vez al arrancar el servidor.
    """
    print("[DB] Inicializando base de datos MySQL...")

    # 1. Crear la base de datos si no existe
    crear_base_datos_si_no_existe()

    # 2. Crear todas las tablas
    sentencias = [s.strip() for s in SQL_CREAR_TABLAS.split(";") if s.strip()]
    for sql in sentencias:
        if sql:
            try:
                db.execute(sql)
            except Exception as e:
                print(f"[DB] Aviso al crear tabla: {e}")

    print("[DB] Tablas creadas/verificadas")

    # 3. Insertar configuración por defecto si no existe
    for clave, valor, desc in CONFIG_DEFAULT:
        db.execute(
            "INSERT IGNORE INTO configuracion (clave, valor, descripcion) VALUES (%s, %s, %s)",
            (clave, valor, desc)
        )

    print("[DB] Configuración por defecto lista")

    # 4. Migraciones seguras (idempotentes — seguro correr en cada arranque)
    migrar_agregar_secuencia_json()

    print("[DB] ✅ Base de datos lista para usar")


# =============================================================================
# HELPERS REUTILIZABLES
# =============================================================================

def get_config(clave: str, default: str = "") -> str:
    """Obtiene un valor de configuración desde la tabla configuracion."""
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
    Importa gestos desde gestos.json a MySQL.
    No sobreescribe gestos ya existentes.
    Respeta secuencias DTW para palabras (campo 'secuencia').
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

        # Para palabras con movimiento, conservar la secuencia DTW
        secuencia_json = None
        if info.get("type") == "word" and info.get("secuencia"):
            secuencia_json = json.dumps(info["secuencia"])

        try:
            db.execute(
                """INSERT IGNORE INTO gestos
                   (nombre, tipo, descripcion, landmarks_json, secuencia_json, muestras_usadas)
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (
                    nombre,
                    info.get("type", "letter"),
                    info.get("description", ""),
                    landmarks_json,
                    secuencia_json,
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
    Exporta todos los gestos de MySQL al archivo gestos.json v3.0.
    - Letras:   campo 'landmarks' (63 floats)
    - Palabras: campo 'secuencia' ([reps][frames][63f]) + 'landmarks' (primer frame)
    El JSON resultante es la fuente de verdad para el reconocedor
    y para sincronizar entre miembros del equipo vía Git.
    """
    rows = db.fetchall("SELECT * FROM gestos ORDER BY nombre")

    gestures = {}
    for row in rows:
        landmarks = json.loads(row["landmarks_json"])
        entry = {
            "name":            row["nombre"],
            "type":            row["tipo"],
            "description":     row["descripcion"] or "",
            "landmarks":       landmarks,
            "muestras_usadas": row["muestras_usadas"],
        }

        # Si es palabra con movimiento, incluir la secuencia DTW completa
        if row["tipo"] == "word" and row.get("secuencia_json"):
            entry["secuencia"] = json.loads(row["secuencia_json"])

        gestures[row["nombre"]] = entry

    total_letras   = sum(1 for r in rows if r["tipo"] != "word")
    total_palabras = sum(1 for r in rows if r["tipo"] == "word")

    data = {
        "version":     "3.0",
        "description": "Base de datos Señas V2 — con DTW para palabras",
        "gestures":    gestures,
        "metadata": {
            "total_gestures": len(gestures),
            "total_letras":   total_letras,
            "total_palabras": total_palabras,
            "exported_at":    datetime.now().isoformat(),
        }
    }

    try:
        with open(ruta_json, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[DB] Exportado → {ruta_json}: {total_letras} letras, {total_palabras} palabras")
        return True
    except Exception as e:
        print(f"[DB] Error exportando: {e}")
        return False