import mysql.connector
import cv2
import mediapipe as mp

# --- CONFIGURACIÓN MEDIA PIPE ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils  # <--- ESTA ES LA LIBRERÍA PARA DIBUJAR
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# --- CONEXIÓN MYSQL ---
try:
    conn = mysql.connector.connect(host="localhost", user="root", password="", database="traductor_gestos")
    cursor = conn.cursor()
    print("Conexión exitosa a MySQL")
except mysql.connector.Error as err:
    print(f"Error de conexión: {err}")
    exit()

cap = cv2.VideoCapture(0)
letra_objetivo = "Z"

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    frame = cv2.flip(frame, 1)
    # Convertir a RGB para MediaPipe
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # --- DIBUJAR LAS LÍNEAS SI DETECTA UNA MANO ---
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Esta línea dibuja los puntos y las conexiones (el esqueleto)
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        # Guardar al presionar ESPACIO
        if cv2.waitKey(1) & 0xFF == ord(' '):
            coords = []
            for lm in results.multi_hand_landmarks[0].landmark:
                coords.extend([lm.x, lm.y, lm.z])

            # Generar nombres de columnas dinámicamente
            col_names = [f"p{i}_{c}" for i in range(21) for c in ['x', 'y', 'z']]
            placeholders = ", ".join(["%s"] * 64)
            sql = f"INSERT INTO gestos (letra, {', '.join(col_names)}) VALUES ({placeholders})"
            cursor.execute(sql, [letra_objetivo] + coords)
            conn.commit()
            print(f"¡Gesto '{letra_objetivo}' guardado!")

    # Mostrar info en pantalla
    cv2.putText(frame, f"Letra: {letra_objetivo}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Captura con Esqueleto", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
conn.close()