"""
interfaz.py
===========
Interfaz gráfica moderna con PyQt5 para el sistema de lenguaje de señas.
Muestra la cámara en tiempo real con overlay de detección,
texto formado y controles de usuario.
"""

import sys                 # Para manejo del sistema
import cv2                 # OpenCV para captura de cámara
import numpy as np         # Para manejo de arrays de imagen
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFrame, QProgressBar, QSizePolicy
)  # Widgets de PyQt5
from PyQt5.QtCore import (
    QTimer, Qt, QThread, pyqtSignal, QObject
)  # Núcleo de PyQt5
from PyQt5.QtGui import (
    QImage, QPixmap, QFont, QPainter, QColor, QPalette
)  # Elementos gráficos
from typing import Optional  # Tipos para anotaciones


class WorkerCamara(QObject):
    """
    Worker que corre en un hilo separado para capturar y procesar frames.
    Emite señales para comunicarse con la interfaz principal.
    """
    # Señal que emite el frame procesado como QImage
    frame_listo = pyqtSignal(QImage)
    
    # Señal para actualizar el estado del reconocimiento
    estado_actualizado = pyqtSignal(dict)
    
    # Señal cuando se detecta una pausa (frase lista para corregir)
    pausa_detectada = pyqtSignal(str)
    
    # Señal de error
    error = pyqtSignal(str)

    def __init__(self, detector, reconocedor, parent=None):
        """
        Inicializa el worker con los módulos de detección y reconocimiento.
        
        Args:
            detector: Instancia de DetectorManos
            reconocedor: Instancia de Reconocedor
        """
        super().__init__(parent)
        self.detector = detector        # Módulo de detección de manos
        self.reconocedor = reconocedor  # Módulo de reconocimiento de gestos
        self._corriendo = False          # Bandera de control
        
        # Captura de cámara
        self.cap: Optional[cv2.VideoCapture] = None
        
        # Contador para procesar frames alternados (optimización)
        self.contador_frames: int = 0

    def iniciar(self) -> None:
        """Inicia la captura de cámara y el loop de procesamiento."""
        self._corriendo = True
        
        # Abrir la cámara (índice 0 = cámara principal)
        self.cap = cv2.VideoCapture(0)
        
        # Verificar que la cámara se abrió correctamente
        if not self.cap.isOpened():
            self.error.emit("No se pudo abrir la cámara. Verifica que esté conectada.")
            return
        
        # Configurar resolución baja para mejor rendimiento
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Ancho 640px
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Alto 480px
        self.cap.set(cv2.CAP_PROP_FPS, 30)            # 30 FPS máximo
        
        # Buffer de 1 frame para minimizar latencia
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        print("[Worker] Cámara iniciada correctamente")
        
        # Iniciar el loop de procesamiento
        self._loop()

    def _loop(self) -> None:
        """
        Loop principal de captura y procesamiento de frames.
        Optimización: procesa 1 de cada 2 frames para reducir carga.
        """
        while self._corriendo:
            # Capturar frame de la cámara
            exito, frame = self.cap.read()
            
            # Si falla la captura, continuar
            if not exito or frame is None:
                continue
            
            # Incrementar contador de frames
            self.contador_frames += 1
            
            # --- Optimización: procesar frames alternados ---
            # Procesar solo el reconocimiento cada 2 frames
            # (la visualización siempre se muestra)
            procesar_reconocimiento = (self.contador_frames % 2 == 0)
            
            # Voltear horizontalmente (efecto espejo, más natural para el usuario)
            frame = cv2.flip(frame, 1)
            
            if procesar_reconocimiento:
                # Detectar manos y obtener resultados de MediaPipe
                frame_procesado, resultados = self.detector.detectar(frame)
                
                # Verificar si hay manos detectadas
                if self.detector.hay_manos(resultados):
                    # Obtener landmarks de la primera mano
                    landmarks = self.detector.obtener_primera_mano(resultados)
                    
                    # Reconocer el gesto con suavizado
                    nombre_gesto, confianza = self.reconocedor.procesar_landmarks(landmarks)
                    
                    # Actualizar el estado del reconocedor
                    self.reconocedor.actualizar_con_mano(nombre_gesto, confianza)
                else:
                    # No hay mano: gestionar pausas
                    self.reconocedor.actualizar_sin_mano()
                    frame_procesado = frame
                
                # Obtener estado actual del reconocimiento
                estado = self.reconocedor.obtener_estado()
                
                # Emitir el estado actualizado a la interfaz
                self.estado_actualizado.emit(estado)
                
                # Verificar si hay una pausa detectada (frase completa)
                frase = self.reconocedor.consumir_pausa()
                if frase:
                    self.pausa_detectada.emit(frase)
            else:
                # Frame sin procesamiento: solo voltear
                frame_procesado = frame
            
            # Convertir el frame a QImage para mostrarlo en PyQt5
            qimage = self._convertir_a_qimage(frame_procesado)
            
            # Emitir el frame a la interfaz
            self.frame_listo.emit(qimage)

    def _convertir_a_qimage(self, frame: np.ndarray) -> QImage:
        """
        Convierte un frame de OpenCV (BGR numpy array) a QImage de PyQt5.
        
        Args:
            frame: Frame BGR de OpenCV
            
        Returns:
            QImage listo para mostrar en un QLabel
        """
        # Obtener dimensiones del frame
        alto, ancho, canales = frame.shape
        
        # Calcular bytes por línea (ancho * 3 canales BGR)
        bytes_por_linea = ancho * canales
        
        # Convertir BGR a RGB (Qt usa RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Crear QImage desde los datos del array
        qimage = QImage(
            frame_rgb.data,          # Datos de la imagen
            ancho,                   # Ancho en píxeles
            alto,                    # Alto en píxeles
            bytes_por_linea,         # Bytes por línea
            QImage.Format_RGB888     # Formato RGB de 8 bits por canal
        )
        
        # Retornar una copia para evitar problemas de memoria
        return qimage.copy()

    def detener(self) -> None:
        """Detiene el loop de captura y libera la cámara."""
        self._corriendo = False
        
        # Liberar la cámara
        if self.cap and self.cap.isOpened():
            self.cap.release()
            print("[Worker] Cámara liberada")


class VentanaPrincipal(QMainWindow):
    """
    Ventana principal de la aplicación.
    Layout moderno con cámara, texto y controles.
    """

    def __init__(self, detector, reconocedor, audio, corrector):
        """
        Inicializa la ventana con todos los módulos del sistema.
        
        Args:
            detector: Instancia de DetectorManos
            reconocedor: Instancia de Reconocedor
            audio: Instancia de GestorAudio
            corrector: Instancia de IaCorrector
        """
        super().__init__()
        
        # Guardar referencias a los módulos
        self.detector = detector
        self.reconocedor = reconocedor
        self.audio = audio
        self.corrector = corrector
        
        # Última frase pronunciada (para no repetir)
        self.ultima_frase_audio = ""
        
        # Configurar la ventana
        self._configurar_ventana()
        
        # Crear y organizar los widgets
        self._crear_interfaz()
        
        # Iniciar el worker de cámara en un hilo separado
        self._iniciar_worker_camara()
        
        print("[Interfaz] Ventana principal creada")

    def _configurar_ventana(self) -> None:
        """Configura propiedades básicas de la ventana."""
        # Título de la ventana
        self.setWindowTitle("🤟 Sistema de Lenguaje de Señas")
        
        # Tamaño inicial de la ventana
        self.resize(1000, 700)
        
        # Centrar la ventana en la pantalla
        self._centrar_ventana()
        
        # Estilo oscuro moderno (stylesheet de Qt)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0d1117;
            }
            QWidget {
                background-color: #0d1117;
                color: #e6edf3;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QLabel {
                color: #e6edf3;
            }
            QPushButton {
                background-color: #21262d;
                color: #e6edf3;
                border: 1px solid #30363d;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #30363d;
                border-color: #58a6ff;
            }
            QPushButton:pressed {
                background-color: #161b22;
            }
            QPushButton#btn_audio_on {
                background-color: #1f6feb;
                border-color: #388bfd;
            }
            QPushButton#btn_audio_off {
                background-color: #21262d;
            }
            QProgressBar {
                border: 1px solid #30363d;
                border-radius: 4px;
                text-align: center;
                background-color: #161b22;
            }
            QProgressBar::chunk {
                background-color: #238636;
                border-radius: 3px;
            }
            QFrame#panel_texto {
                background-color: #161b22;
                border: 1px solid #30363d;
                border-radius: 8px;
                padding: 8px;
            }
            QFrame#panel_camara {
                border: 2px solid #30363d;
                border-radius: 8px;
            }
        """)

    def _centrar_ventana(self) -> None:
        """Centra la ventana en la pantalla."""
        from PyQt5.QtWidgets import QDesktopWidget
        pantalla = QDesktopWidget().screenGeometry()
        ventana = self.geometry()
        x = (pantalla.width() - ventana.width()) // 2
        y = (pantalla.height() - ventana.height()) // 2
        self.move(x, y)

    def _crear_interfaz(self) -> None:
        """Crea y organiza todos los widgets de la interfaz."""
        # Widget central
        central = QWidget()
        self.setCentralWidget(central)
        
        # Layout principal: horizontal (cámara izquierda | panel derecho)
        layout_principal = QHBoxLayout(central)
        layout_principal.setContentsMargins(12, 12, 12, 12)
        layout_principal.setSpacing(12)
        
        # --- Panel izquierdo: Vista de cámara ---
        panel_camara = self._crear_panel_camara()
        layout_principal.addWidget(panel_camara, stretch=2)  # 2/3 del espacio
        
        # --- Panel derecho: Texto + Controles ---
        panel_control = self._crear_panel_control()
        layout_principal.addWidget(panel_control, stretch=1)  # 1/3 del espacio

    def _crear_panel_camara(self) -> QFrame:
        """
        Crea el panel de visualización de la cámara.
        
        Returns:
            QFrame con el QLabel de la cámara
        """
        # Frame contenedor con borde estilizado
        frame = QFrame()
        frame.setObjectName("panel_camara")
        
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        
        # Título del panel
        titulo = QLabel("📷 Cámara en tiempo real")
        titulo.setStyleSheet("font-size: 13px; color: #8b949e; padding: 2px;")
        titulo.setAlignment(Qt.AlignCenter)
        layout.addWidget(titulo)
        
        # Label donde se muestra el video
        self.label_camara = QLabel()
        self.label_camara.setMinimumSize(480, 360)  # Tamaño mínimo
        self.label_camara.setAlignment(Qt.AlignCenter)
        self.label_camara.setStyleSheet("""
            background-color: #010409;
            border-radius: 4px;
            color: #8b949e;
        """)
        self.label_camara.setText("Iniciando cámara...")
        self.label_camara.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.label_camara)
        
        return frame

    def _crear_panel_control(self) -> QWidget:
        """
        Crea el panel derecho con texto e información del reconocimiento.
        
        Returns:
            QWidget con todos los controles
        """
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        
        # --- Sección: Letra actual ---
        layout.addWidget(self._crear_seccion_letra())
        
        # --- Sección: Confianza ---
        layout.addWidget(self._crear_seccion_confianza())
        
        # --- Sección: Palabra en formación ---
        layout.addWidget(self._crear_seccion_palabra())
        
        # --- Sección: Frase completa ---
        layout.addWidget(self._crear_seccion_frase())
        
        # Espacio flexible
        layout.addStretch()
        
        # --- Sección: Botones de control ---
        layout.addWidget(self._crear_seccion_botones())
        
        # --- Etiqueta de estado ---
        self.label_estado = QLabel("Sistema listo")
        self.label_estado.setStyleSheet("""
            color: #8b949e;
            font-size: 11px;
            padding: 4px;
        """)
        self.label_estado.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label_estado)
        
        return panel

    def _crear_seccion_letra(self) -> QFrame:
        """Crea la sección que muestra la letra detectada actualmente."""
        frame = QFrame()
        frame.setObjectName("panel_texto")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)
        
        # Etiqueta del título
        titulo = QLabel("GESTO ACTUAL")
        titulo.setStyleSheet("color: #8b949e; font-size: 10px; letter-spacing: 2px;")
        layout.addWidget(titulo)
        
        # Etiqueta grande que muestra la letra
        self.label_letra = QLabel("—")
        self.label_letra.setStyleSheet("""
            color: #58a6ff;
            font-size: 52px;
            font-weight: bold;
            padding: 4px;
        """)
        self.label_letra.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label_letra)
        
        return frame

    def _crear_seccion_confianza(self) -> QFrame:
        """Crea la barra de progreso de confianza del gesto."""
        frame = QFrame()
        frame.setObjectName("panel_texto")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)
        
        # Título
        titulo = QLabel("CONFIANZA")
        titulo.setStyleSheet("color: #8b949e; font-size: 10px; letter-spacing: 2px;")
        layout.addWidget(titulo)
        
        # Barra de progreso
        self.barra_confianza = QProgressBar()
        self.barra_confianza.setMinimum(0)
        self.barra_confianza.setMaximum(100)
        self.barra_confianza.setValue(0)
        self.barra_confianza.setFixedHeight(20)
        layout.addWidget(self.barra_confianza)
        
        return frame

    def _crear_seccion_palabra(self) -> QFrame:
        """Crea la sección que muestra la palabra en formación."""
        frame = QFrame()
        frame.setObjectName("panel_texto")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)
        
        # Título
        titulo = QLabel("PALABRA EN FORMACIÓN")
        titulo.setStyleSheet("color: #8b949e; font-size: 10px; letter-spacing: 2px;")
        layout.addWidget(titulo)
        
        # Texto de la palabra
        self.label_palabra = QLabel("")
        self.label_palabra.setStyleSheet("""
            color: #3fb950;
            font-size: 24px;
            font-weight: bold;
            padding: 4px;
            min-height: 40px;
        """)
        self.label_palabra.setAlignment(Qt.AlignCenter)
        self.label_palabra.setWordWrap(True)
        layout.addWidget(self.label_palabra)
        
        return frame

    def _crear_seccion_frase(self) -> QFrame:
        """Crea la sección que muestra la frase completa."""
        frame = QFrame()
        frame.setObjectName("panel_texto")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)
        
        # Título
        titulo = QLabel("FRASE COMPLETA")
        titulo.setStyleSheet("color: #8b949e; font-size: 10px; letter-spacing: 2px;")
        layout.addWidget(titulo)
        
        # Texto de la frase
        self.label_frase = QLabel("")
        self.label_frase.setStyleSheet("""
            color: #e6edf3;
            font-size: 16px;
            padding: 8px;
            min-height: 60px;
            background-color: #010409;
            border-radius: 4px;
            line-height: 1.5;
        """)
        self.label_frase.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.label_frase.setWordWrap(True)   # Envolver texto largo
        layout.addWidget(self.label_frase)
        
        return frame

    def _crear_seccion_botones(self) -> QWidget:
        """Crea los botones de control."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        # Botón: Activar/desactivar audio
        self.btn_audio = QPushButton("🔊 Audio: ON")
        self.btn_audio.setObjectName("btn_audio_on")
        self.btn_audio.setFixedHeight(40)
        self.btn_audio.clicked.connect(self._toggle_audio)
        layout.addWidget(self.btn_audio)
        
        # Botón: Limpiar texto
        btn_limpiar = QPushButton("🗑️ Limpiar texto")
        btn_limpiar.setFixedHeight(40)
        btn_limpiar.clicked.connect(self._limpiar_texto)
        layout.addWidget(btn_limpiar)
        
        # Botón: Reproducir frase actual
        btn_reproducir = QPushButton("▶️ Reproducir frase")
        btn_reproducir.setFixedHeight(40)
        btn_reproducir.clicked.connect(self._reproducir_frase)
        layout.addWidget(btn_reproducir)
        
        # Botón: Finalizar palabra (forzar pausa)
        btn_fin_palabra = QPushButton("⏸️ Finalizar palabra")
        btn_fin_palabra.setFixedHeight(40)
        btn_fin_palabra.clicked.connect(self._forzar_fin_palabra)
        layout.addWidget(btn_fin_palabra)
        
        # Botón: Salir
        btn_salir = QPushButton("❌ Salir")
        btn_salir.setFixedHeight(40)
        btn_salir.setStyleSheet("""
            QPushButton {
                background-color: #da3633;
                border-color: #f85149;
                color: white;
            }
            QPushButton:hover {
                background-color: #b62324;
            }
        """)
        btn_salir.clicked.connect(self.close)
        layout.addWidget(btn_salir)
        
        return widget

    def _iniciar_worker_camara(self) -> None:
        """Crea e inicia el worker de cámara en un hilo separado."""
        # Crear hilo de Qt
        self.hilo_camara = QThread()
        
        # Crear el worker
        self.worker = WorkerCamara(self.detector, self.reconocedor)
        
        # Mover el worker al hilo
        self.worker.moveToThread(self.hilo_camara)
        
        # Conectar señales del worker a slots de la interfaz
        self.worker.frame_listo.connect(self._actualizar_frame)
        self.worker.estado_actualizado.connect(self._actualizar_estado)
        self.worker.pausa_detectada.connect(self._manejar_pausa)
        self.worker.error.connect(self._mostrar_error)
        
        # Iniciar el worker cuando el hilo empiece
        self.hilo_camara.started.connect(self.worker.iniciar)
        
        # Iniciar el hilo
        self.hilo_camara.start()

    def _actualizar_frame(self, qimage: QImage) -> None:
        """
        Actualiza el QLabel de la cámara con el nuevo frame.
        
        Args:
            qimage: El frame como QImage
        """
        # Escalar la imagen al tamaño del label manteniendo proporción
        pixmap = QPixmap.fromImage(qimage)
        pixmap_escalado = pixmap.scaled(
            self.label_camara.size(),        # Tamaño del label
            Qt.KeepAspectRatio,              # Mantener proporción
            Qt.SmoothTransformation          # Suavizado
        )
        self.label_camara.setPixmap(pixmap_escalado)

    def _actualizar_estado(self, estado: dict) -> None:
        """
        Actualiza los elementos de texto con el estado del reconocimiento.
        
        Args:
            estado: Diccionario con el estado actual
        """
        # Actualizar letra actual
        letra = estado.get("letra_actual", "")
        self.label_letra.setText(letra if letra else "—")
        
        # Actualizar barra de confianza
        confianza = estado.get("confianza", 0.0)
        self.barra_confianza.setValue(int(confianza * 100))
        
        # Cambiar color según confianza
        if confianza > 0.7:
            estilo_barra = "QProgressBar::chunk { background-color: #238636; }"
        elif confianza > 0.4:
            estilo_barra = "QProgressBar::chunk { background-color: #d29922; }"
        else:
            estilo_barra = "QProgressBar::chunk { background-color: #da3633; }"
        self.barra_confianza.setStyleSheet(estilo_barra)
        
        # Actualizar palabra en formación
        palabra = estado.get("palabra_actual", "")
        self.label_palabra.setText(palabra)
        
        # Actualizar frase completa
        frase = estado.get("frase_completa", "")
        self.label_frase.setText(frase)

    def _manejar_pausa(self, frase: str) -> None:
        """
        Maneja cuando se detecta una pausa (frase completa lista).
        Activa la corrección de IA y el audio.
        
        Args:
            frase: La frase completa para corregir
        """
        if not frase:
            return
        
        self.label_estado.setText("✨ Corrigiendo frase...")
        
        # Aplicar corrección con callback asíncrono
        frase_corregida = self.corrector.corregir(
            frase,
            confianza=self.reconocedor.confianza_actual,
            callback=self._recibir_correccion_ia
        )
        
        # Actualizar la frase con la corrección local inmediata
        self.reconocedor.frase_completa = frase_corregida
        self.label_frase.setText(frase_corregida)
        
        # Reproducir la frase corregida
        if frase_corregida != self.ultima_frase_audio:
            self.audio.hablar_frase(frase_corregida)
            self.ultima_frase_audio = frase_corregida
        
        self.label_estado.setText("✅ Frase reproducida")

    def _recibir_correccion_ia(self, frase_ia: str) -> None:
        """
        Callback que recibe la corrección de IA cuando termina (asíncrono).
        
        Args:
            frase_ia: Frase corregida por la IA
        """
        # Actualizar la frase con la corrección de IA (mejor que la local)
        self.reconocedor.frase_completa = frase_ia
        self.label_frase.setText(frase_ia)
        self.label_estado.setText("🤖 Frase corregida con IA")
        
        # Reproducir la versión mejorada
        self.audio.hablar_frase(frase_ia)

    def _toggle_audio(self) -> None:
        """Activa o desactiva el audio."""
        activo = self.audio.toggle()
        
        if activo:
            self.btn_audio.setText("🔊 Audio: ON")
            self.btn_audio.setObjectName("btn_audio_on")
        else:
            self.btn_audio.setText("🔇 Audio: OFF")
            self.btn_audio.setObjectName("btn_audio_off")
        
        # Forzar actualización del estilo
        self.btn_audio.setStyle(self.btn_audio.style())

    def _limpiar_texto(self) -> None:
        """Limpia todo el texto del reconocedor y la interfaz."""
        self.reconocedor.limpiar_todo()
        
        # Limpiar los labels
        self.label_letra.setText("—")
        self.label_palabra.setText("")
        self.label_frase.setText("")
        self.barra_confianza.setValue(0)
        self.ultima_frase_audio = ""
        
        self.label_estado.setText("Texto limpiado")

    def _reproducir_frase(self) -> None:
        """Reproduce la frase actual manualmente."""
        frase = self.reconocedor.frase_completa
        if frase:
            self.audio.hablar_frase(frase)
            self.label_estado.setText("🔊 Reproduciendo...")
        else:
            self.label_estado.setText("No hay frase para reproducir")

    def _forzar_fin_palabra(self) -> None:
        """Fuerza el final de la palabra actual."""
        self.reconocedor.forzar_fin_palabra()
        self.label_estado.setText("⏸️ Palabra finalizada manualmente")

    def _mostrar_error(self, mensaje: str) -> None:
        """
        Muestra un error en la interfaz.
        
        Args:
            mensaje: Mensaje de error a mostrar
        """
        self.label_estado.setText(f"❌ Error: {mensaje}")
        self.label_camara.setText(f"Error: {mensaje}")
        print(f"[Interfaz] Error: {mensaje}")

    def closeEvent(self, event) -> None:
        """
        Maneja el cierre de la ventana.
        Libera todos los recursos antes de cerrar.
        """
        print("[Interfaz] Cerrando aplicación...")
        
        # Detener el worker de cámara
        if hasattr(self, 'worker'):
            self.worker.detener()
        
        # Detener el hilo de cámara
        if hasattr(self, 'hilo_camara'):
            self.hilo_camara.quit()
            self.hilo_camara.wait(2000)  # Esperar máximo 2 segundos
        
        # Liberar detector de MediaPipe
        self.detector.liberar()
        
        # Liberar audio
        self.audio.liberar()
        
        print("[Interfaz] Recursos liberados. ¡Hasta luego!")
        
        # Aceptar el evento de cierre
        event.accept()
