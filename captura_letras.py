import cv2
import mediapipe as mp
import joblib
import numpy as np
import pyttsx3
from collections import deque

# --- CONFIGURACIÓN ---
try:
    model = joblib.load('modelo_final.pkl')
except:
    print("Modelo no encontrado.");
    exit()

engine = pyttsx3.init()
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.85,  # Más riguroso
    min_tracking_confidence=0.85
)

# --- MOTORES DE ESTABILIDAD ---
# 1. Suavizado de coordenadas (Filtro Pasa-Bajos)
ALPHA = 0.18  # Cuanto menor sea, más "pesada" y quieta estará la mano
coords_suavizadas = None

# 2. Persistencia de Predicción (Búfer de decisión)
# Guarda las últimas 10 predicciones para elegir la más frecuente
buffer_predicciones = deque(maxlen=10)

# --- DISEÑO ---
ANCHO_MENU, ANCHO_VIDEO, ALTO = 300, 640, 480
COLOR_PANEL, COLOR_ACENTO = (25, 25, 25), (0, 255, 127)
letra_confirmada = "..."

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    frame = cv2.flip(cv2.resize(frame, (ANCHO_VIDEO, ALTO)), 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extraer coordenadas actuales
            coords_raw = []
            for lm in hand_landmarks.landmark:
                coords_raw.extend([lm.x, lm.y, lm.z])
            coords_raw = np.array(coords_raw)

            # FILTRO 1: SUAVIZADO DE MOVIMIENTO
            if coords_suavizadas is None:
                coords_suavizadas = coords_raw
            else:
                coords_suavizadas = ALPHA * coords_raw + (1 - ALPHA) * coords_suavizadas

            # FILTRO 2: ESTABILIDAD DE PREDICCIÓN
            pred = model.predict([coords_suavizadas])[0]
            buffer_predicciones.append(pred)

            # Elegimos la letra que más se repite en el búfer (Moda)
            letra_mas_frecuente = max(set(buffer_predicciones), key=list(buffer_predicciones).count)

            # Solo hablamos si la letra es ultra-estable (ej. 9 de 10 veces en el búfer)
            if list(buffer_predicciones).count(letra_mas_frecuente) >= 8:
                if letra_mas_frecuente != letra_confirmada:
                    letra_confirmada = letra_mas_frecuente
                    engine.say(letra_confirmada);
                    engine.runAndWait()

            # Dibujo de landmarks suavizados
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # --- RENDERIZADO (UI) ---
    canvas = np.zeros((ALTO, ANCHO_MENU + ANCHO_VIDEO, 3), dtype=np.uint8)
    canvas[:, :ANCHO_MENU] = COLOR_PANEL
    canvas[:, ANCHO_MENU:] = frame

    cv2.putText(canvas, "SIGNFLOW PRO", (40, 60), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(canvas, letra_confirmada, (110, 320), cv2.FONT_HERSHEY_DUPLEX, 4, (255, 255, 255), 8)

    # Barra de confianza visual basada en el búfer
    confianza_buffer = (list(buffer_predicciones).count(letra_confirmada) / 10) * 220
    cv2.rectangle(canvas, (40, 350), (260, 355), (60, 60, 60), -1)
    cv2.rectangle(canvas, (40, 350), (40 + int(confianza_buffer), 355), COLOR_ACENTO, -1)

    cv2.imshow("SignFlow Studio Professional", canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()