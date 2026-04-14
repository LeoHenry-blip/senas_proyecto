import cv2
import time


def test_camara_rendimiento():
    # 0 suele ser la webcam integrada, 1 o 2 suele ser Iriun/DroidCam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: No se pudo acceder a la cámara.")
        return

    print("--- Test de Rendimiento Iniciado ---")
    print("Presiona 'q' para salir.")

    prev_time = 0

    while True:
        start_frame = time.time()  # Inicio de procesamiento

        success, frame = cap.read()
        if not success:
            break

        # Calcular FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        # Calcular Latencia (ms)
        latencia = (time.time() - start_frame) * 1000

        # Dibujar info en pantalla
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Rendimiento: {latencia:.2f} ms", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Test de Camara - Proyecto Gestos", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_camara_rendimiento()