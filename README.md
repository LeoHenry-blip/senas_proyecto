# 🤟 Señas V2 — Sistema Web de Lenguaje de Señas en Tiempo Real

Sistema web multiusuario para comunicación entre personas sordas y oyentes mediante
reconocimiento de lenguaje de señas, chat en tiempo real y videollamadas.

---

## 🏗 Estructura del proyecto

```
señas_v2/
├── backend/
│   ├── main.py                  # Servidor FastAPI principal
│   ├── requirements.txt
│   ├── .env.example
│   ├── gestos.json              # Copiar desde sistema V1
│   ├── db/
│   │   └── database.py          # Conexión MySQL + creación de tablas
│   ├── core/
│   │   └── auth.py              # JWT, bcrypt, login, OAuth
│   ├── routes/
│   │   ├── auth.py              # /api/auth/*
│   │   ├── reuniones.py         # /api/reuniones/*
│   │   ├── admin.py             # /api/admin/*
│   │   └── landmarks.py         # /api/admin/extraer-landmarks
│   ├── websocket/
│   │   ├── manager.py           # Gestor de salas WebSocket
│   │   └── endpoint.py          # Handler principal /ws/{sala_id}
│   │
│   │   # Módulos reutilizados del sistema V1:
│   ├── detector_manos.py
│   ├── reconocedor.py
│   ├── base_datos.py
│   ├── audio.py
│   └── ia_corrector.py
│
└── frontend/
    ├── index.html               # Login / Registro
    ├── dashboard.html           # Pantalla principal post-login
    ├── reunion.html             # Sala de reunión (3 paneles)
    ├── admin.html               # Panel de administración
    ├── css/
    │   └── main.css             # Sistema de diseño dark mode
    └── js/
        └── app.js               # API client, WS, WebRTC, utilidades
```

---

## ⚡ Instalación

### 1. Requisitos previos

- Python 3.10+
- MySQL 8.0+ (o MariaDB 10.6+)
- Node.js (opcional, solo si se quiere un bundler)

### 2. Configurar MySQL

```sql
-- Conectar a MySQL como root y crear el usuario del sistema
CREATE USER 'senas'@'localhost' IDENTIFIED BY 'tu_password';
GRANT ALL PRIVILEGES ON senas_v2.* TO 'senas'@'localhost';
FLUSH PRIVILEGES;
```

> La base de datos `senas_v2` y todas las tablas se crean **automáticamente** al arrancar el servidor.

### 3. Configurar variables de entorno

```bash
cd backend
cp .env.example .env
# Editar .env con tus credenciales de MySQL
```

### 4. Instalar dependencias

```bash
cd backend
pip install -r requirements.txt
```

### 5. Copiar módulos del sistema V1

```bash
# Copiar los módulos Python del sistema original
cp /ruta/v1/detector_manos.py  backend/
cp /ruta/v1/reconocedor.py     backend/
cp /ruta/v1/base_datos.py      backend/
cp /ruta/v1/audio.py           backend/
cp /ruta/v1/ia_corrector.py    backend/
cp /ruta/v1/gestos.json        backend/
```

### 6. Arrancar el servidor

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 7. Acceder al sistema

- **Aplicación web:** http://localhost:8000
- **Swagger API:** http://localhost:8000/api/docs

---

## 👤 Crear el primer usuario administrador

Al arrancar el servidor por primera vez, regístrate desde la interfaz web
y luego asigna el rol de admin directamente en MySQL:

```sql
UPDATE usuarios SET rol = 'admin' WHERE email = 'tu@email.com';
```

O usando el endpoint de la API (con token de otro admin):

```bash
curl -X PATCH http://localhost:8000/api/admin/usuarios/1 \
  -H "Authorization: Bearer TU_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"rol": "admin"}'
```

---

## 🖥 Flujo de uso

### Usuario sordomudo

1. Iniciar sesión → Dashboard
2. Crear o unirse a una reunión
3. Activar la cámara en el panel izquierdo
4. Realizar señas → el sistema reconoce letras/palabras automáticamente
5. Las frases se envían al chat cuando se detecta una pausa
6. Los demás participantes ven los subtítulos en tiempo real y escuchan el audio

### Usuario oyente

1. Unirse a la reunión con el código
2. Ver el chat con los mensajes traducidos
3. Escribir mensajes de texto en el input inferior

### Administrador

1. Acceder al **Panel Admin** desde el dashboard
2. **Entrenador de gestos**: activar cámara → escribir nombre del gesto → capturar
3. Los gestos se guardan en MySQL y se exportan automáticamente a `gestos.json`
4. El reconocedor se recarga en tiempo real sin reiniciar el servidor

---

## 🔌 API WebSocket

Conectar al WebSocket de una sala:
```
ws://localhost:8000/ws/CODIGO-SALA?token=JWT_TOKEN
```

### Mensajes del cliente al servidor

| tipo | descripción | campos adicionales |
|------|-------------|-------------------|
| `frame` | Frame de cámara en base64 | `data: string` |
| `mensaje_texto` | Mensaje de chat escrito | `texto: string` |
| `fin_frase` | Forzar cierre de palabra | — |
| `limpiar` | Limpiar reconocedor | — |
| `webrtc_señal` | Señal WebRTC | `para: int, señal: object` |

### Mensajes del servidor al cliente

| tipo | descripción |
|------|-------------|
| `traduccion_live` | Estado del reconocimiento en tiempo real |
| `mensaje_chat` | Nuevo mensaje confirmado en el chat |
| `mensaje_corregido_ia` | Actualización del texto corregido por IA |
| `sistema` | Notificación del sistema (entró/salió alguien) |
| `sala_info` | Lista de participantes al conectarse |
| `webrtc_señal` | Señal WebRTC reenviada entre peers |

---

## 🗄 Esquema MySQL

| Tabla | Descripción |
|-------|-------------|
| `usuarios` | Usuarios registrados (local + Google OAuth) |
| `reuniones` | Salas de videollamada y chat |
| `reunion_participantes` | Historial de quién estuvo en cada sala |
| `mensajes` | Todos los mensajes con texto original y corregido |
| `gestos` | Base de datos de gestos entrenados |
| `configuracion` | Parámetros del reconocedor (umbral, FPS, etc.) |
| `sesiones` | Tokens JWT activos |

---

## ⚙️ Variables de configuración del reconocedor

Ajustables desde el Panel Admin sin reiniciar el servidor:

| Clave | Por defecto | Descripción |
|-------|-------------|-------------|
| `umbral_similitud` | 0.35 | Máxima distancia euclidiana para reconocer |
| `frames_suavizado` | 10 | Buffer de frames para suavizado |
| `confianza_minima` | 0.40 | Confianza mínima aceptada |
| `tiempo_confirmacion` | 1.2s | Tiempo que debe sostenerse el gesto |
| `tiempo_pausa_letra` | 1.5s | Pausa para separar letras |
| `tiempo_pausa_palabra` | 3.0s | Pausa para cerrar una palabra |
| `velocidad_voz` | 150 | WPM de la síntesis de voz |

---

## 🔒 Seguridad

- Contraseñas hasheadas con **bcrypt (rounds=12)**
- Autenticación mediante **JWT** (expiración configurable)
- Rutas del panel admin protegidas con rol `admin`
- WebSocket autenticado con token en query parameter
- CORS configurable (restringir en producción)

---

## 🚀 Despliegue en producción

```bash
# Con múltiples workers
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4

# Con Gunicorn + Uvicorn workers
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

> **Nota:** Los WebSockets de reconocimiento son stateful (cada usuario tiene su propio
> detector MediaPipe). Con múltiples workers se recomienda usar sticky sessions o
> un gestor de estado compartido como Redis para las salas.
