"""
audio.py
========
Módulo de síntesis de voz usando pyttsx3 (offline).
Reproduce letras, palabras y frases en un hilo separado
para no bloquear la interfaz gráfica.
"""

import threading    # Para reproducción en hilo separado
import queue        # Cola para gestionar solicitudes de audio
from typing import Optional  # Tipos para anotaciones

# Importar pyttsx3 con manejo de error si no está instalado
try:
    import pyttsx3
    PYTTSX3_DISPONIBLE = True
except ImportError:
    PYTTSX3_DISPONIBLE = False
    print("[Audio] ADVERTENCIA: pyttsx3 no instalado. Sin audio.")


class GestorAudio:
    """
    Gestiona la reproducción de texto a voz de forma asíncrona.
    Usa una cola para evitar que las solicitudes de audio se pisen.
    El motor de voz corre en su propio hilo para no bloquear la UI.
    """

    def __init__(self, velocidad: int = 150, volumen: float = 1.0):
        """
        Inicializa el motor de voz y el hilo de reproducción.
        
        Args:
            velocidad: Palabras por minuto (100-200 recomendado)
            volumen: Volumen de 0.0 a 1.0
        """
        # Estado de activación del audio
        self.activo: bool = True
        
        # Cola de tareas de audio (thread-safe)
        # maxsize=3 evita que se acumule demasiado en la cola
        self.cola_audio: queue.Queue = queue.Queue(maxsize=3)
        
        # Motor pyttsx3 (None si no está disponible)
        self.motor: Optional[object] = None
        
        # Hilo de reproducción
        self.hilo: Optional[threading.Thread] = None
        
        # Bandera para detener el hilo al cerrar la app
        self._ejecutando: bool = True
        
        # Inicializar solo si pyttsx3 está disponible
        if PYTTSX3_DISPONIBLE:
            self._inicializar_motor(velocidad, volumen)
            self._iniciar_hilo()
        else:
            print("[Audio] Funcionando sin audio (pyttsx3 no disponible)")

    def _inicializar_motor(self, velocidad: int, volumen: float) -> None:
        """
        Crea e inicializa el motor pyttsx3 con los parámetros dados.
        
        Args:
            velocidad: WPM del habla
            volumen: Nivel de volumen
        """
        try:
            # Crear instancia del motor de voz
            self.motor = pyttsx3.init()
            
            # Configurar velocidad de habla (palabras por minuto)
            self.motor.setProperty('rate', velocidad)
            
            # Configurar volumen (0.0 a 1.0)
            self.motor.setProperty('volume', volumen)
            
            # Intentar usar voz en español si está disponible
            voces = self.motor.getProperty('voices')
            for voz in voces:
                # Buscar una voz en español
                if 'spanish' in voz.name.lower() or 'es' in voz.id.lower():
                    self.motor.setProperty('voice', voz.id)
                    print(f"[Audio] Voz en español: {voz.name}")
                    break
            else:
                print("[Audio] Usando voz por defecto del sistema")
            
            print(f"[Audio] Motor inicializado (vel={velocidad}, vol={volumen})")
            
        except Exception as e:
            print(f"[Audio] Error al inicializar motor: {e}")
            self.motor = None

    def _iniciar_hilo(self) -> None:
        """
        Inicia el hilo dedicado a procesar la cola de audio.
        """
        # Crear hilo como daemon para que se cierre con la app
        self.hilo = threading.Thread(
            target=self._loop_audio,
            name="HiloAudio",
            daemon=True  # Se cierra automáticamente cuando la app termina
        )
        self.hilo.start()
        print("[Audio] Hilo de audio iniciado")

    def _loop_audio(self) -> None:
        """
        Bucle principal del hilo de audio.
        Espera textos en la cola y los reproduce uno por uno.
        """
        while self._ejecutando:
            try:
                # Esperar texto en la cola (timeout para poder verificar _ejecutando)
                texto = self.cola_audio.get(timeout=0.5)
                
                # Verificar que el motor esté disponible y el audio activo
                if self.motor is not None and self.activo and self._ejecutando:
                    try:
                        # Reproducir el texto en voz
                        self.motor.say(texto)
                        self.motor.runAndWait()
                    except Exception as e:
                        # Error durante reproducción, continuar
                        print(f"[Audio] Error al reproducir: {e}")
                
                # Marcar tarea como completada en la cola
                self.cola_audio.task_done()
                
            except queue.Empty:
                # La cola está vacía, seguir esperando
                continue
            except Exception as e:
                print(f"[Audio] Error en hilo de audio: {e}")

    def hablar(self, texto: str, prioridad: bool = False) -> None:
        """
        Solicita que se reproduzca un texto.
        No bloquea: la reproducción ocurre en el hilo de audio.
        
        Args:
            texto: Texto a reproducir
            prioridad: Si es True, vacía la cola antes de agregar este texto
        """
        # No hacer nada si el audio está desactivado
        if not self.activo:
            return
        
        # No hacer nada si el texto está vacío
        if not texto or not texto.strip():
            return
        
        # Si es prioridad, limpiar cola primero
        if prioridad:
            self._limpiar_cola()
        
        # Intentar agregar a la cola sin bloquear
        try:
            # put_nowait no bloquea: si la cola está llena, descarta el texto
            self.cola_audio.put_nowait(texto)
        except queue.Full:
            # Cola llena: descartar este texto para no bloquear
            # (es normal en reconocimiento continuo)
            pass

    def hablar_letra(self, letra: str) -> None:
        """
        Reproduce una letra individual (opcional, puede ser molesto).
        
        Args:
            letra: La letra a reproducir
        """
        if letra and len(letra) == 1:
            self.hablar(letra)

    def hablar_palabra(self, palabra: str) -> None:
        """
        Reproduce una palabra completa con prioridad.
        
        Args:
            palabra: La palabra a reproducir
        """
        if palabra:
            self.hablar(palabra, prioridad=True)

    def hablar_frase(self, frase: str) -> None:
        """
        Reproduce una frase completa con máxima prioridad.
        Limpia la cola para que esta frase se reproduzca de inmediato.
        
        Args:
            frase: La frase a reproducir
        """
        if frase:
            self.hablar(frase, prioridad=True)

    def _limpiar_cola(self) -> None:
        """
        Vacía la cola de audio pendiente.
        Útil para interrumpir reproducciones antiguas.
        """
        while not self.cola_audio.empty():
            try:
                self.cola_audio.get_nowait()
                self.cola_audio.task_done()
            except queue.Empty:
                break

    def activar(self) -> None:
        """Activa la reproducción de audio."""
        self.activo = True
        print("[Audio] Audio activado")

    def desactivar(self) -> None:
        """Desactiva la reproducción de audio."""
        self.activo = False
        self._limpiar_cola()  # Limpiar pendientes
        print("[Audio] Audio desactivado")

    def toggle(self) -> bool:
        """
        Alterna entre activado y desactivado.
        
        Returns:
            True si quedó activado, False si quedó desactivado
        """
        if self.activo:
            self.desactivar()
        else:
            self.activar()
        return self.activo

    def cambiar_velocidad(self, velocidad: int) -> None:
        """
        Cambia la velocidad de habla en tiempo de ejecución.
        
        Args:
            velocidad: Nueva velocidad en WPM
        """
        if self.motor:
            self.motor.setProperty('rate', velocidad)
            print(f"[Audio] Velocidad cambiada a {velocidad} WPM")

    def liberar(self) -> None:
        """
        Detiene el hilo de audio y libera recursos.
        Llamar siempre al cerrar la aplicación.
        """
        self._ejecutando = False  # Señal para detener el hilo
        
        # Limpiar la cola para desbloquear el hilo si está esperando
        self._limpiar_cola()
        
        # Esperar a que el hilo termine (con timeout)
        if self.hilo and self.hilo.is_alive():
            self.hilo.join(timeout=2.0)
        
        # Detener el motor de voz
        if self.motor:
            try:
                self.motor.stop()
            except Exception:
                pass
        
        print("[Audio] Recursos de audio liberados")
