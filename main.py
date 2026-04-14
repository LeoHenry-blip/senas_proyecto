import cv2
import mediapipe as mp
import joblib
import pyttsx3

# 1. Cargar el "cerebro" generado por entrenar.py
try:
    model = joblib.load('modelo_final.pkl')
    print("Modelo cargado. ¡Listo para traducir!")
except:
    print("Error: No existe 'modelo_final.pkl'. Ejecuta primero entrenar.py")
    exit()

engine = pyttsx3.init()
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

contador_letras = 0
ultima_letra_detectada = None
letra_confirmada = "Esperando..."
UMBRAL_CONFIRMACION = 15  # Cuántos frames debe estar segura la IA

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    frame = cv2.flip(frame, 1)
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extraer coordenadas para predecir
            coords = []
            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])

            # Predicción basada en tu base de datos
            prediccion = model.predict([coords])
            letra_actual = prediccion[0]

            # Lógica de confirmación
            if letra_actual == ultima_letra_detectada:
                contador_letras += 1
            else:
                contador_letras = 0
                ultima_letra_detectada = letra_actual

            if contador_letras >= UMBRAL_CONFIRMACION:
                if letra_actual != letra_confirmada:
                    letra_confirmada = letra_actual
                    engine.say(letra_confirmada)
                    engine.runAndWait()
                contador_letras = 0

    # Interfaz
    cv2.putText(frame, f"Traduccion: {letra_confirmada}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.imshow("Traductor en Tiempo Real", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()