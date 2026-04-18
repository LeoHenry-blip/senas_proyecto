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
    conn = mysql.connector.connect(host="localhost", user="root", password="", database="senas_v2")
    cursor = conn.cursor()
    print("Conexión exitosa a MySQL")
except mysql.connector.Error as err:
    print(f"Error de conexión: {err}")
    exit()

cap = cv2.VideoCapture(0)
letra_objetivo = "Z"
contador_muestras = 0

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
        if cv2.waitKey(1) & 0xFF == ord('s'):
            coords = []
            for lm in results.multi_hand_landmarks[0].landmark:
                coords.extend([lm.x, lm.y, lm.z])

            # Generar nombres de columnas dinámicamente
            col_names = [f"p{i}_{c}" for i in range(21) for c in ['x', 'y', 'z']]
            placeholders = ", ".join(["%s"] * 64)
            sql = f"INSERT INTO gestos (letra, {', '.join(col_names)}) VALUES ({placeholders})"

            try:
                cursor.execute(sql,[letra_objetivo]+coords)
                conn.commit()
                contador_muestras += 1
                print(f"Muestra #{contador_muestras} guardada para {letra_objetivo}")
            except Exception as e:
                print(f"Error al guardar porque: {e}")


    # Mostrar info en pantalla
    cv2.rectangle(frame,(0,0),(300,50),(0,0,0),-1)
    cv2.putText(frame, f"Gesto: {letra_objetivo}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    cv2.putText(frame, f"Muestras: {contador_muestras}",(10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)

    cv2.imshow("Captura masiva de frames", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
conn.close()