"""
ia_corrector.py
===============
Módulo de corrección inteligente de frases generadas por el sistema de señas.
Implementa corrección híbrida en 2 niveles:
  - Nivel 1: Corrección local offline (diccionario + reglas)
  - Nivel 2: Corrección con IA externa (API opcional)
"""

import re           # Expresiones regulares para corrección de texto
import threading    # Para ejecutar corrección en hilo separado
from typing import Optional, Callable  # Tipos para anotaciones


# =============================================================================
# DICCIONARIO DE ERRORES COMUNES
# Mapea errores frecuentes del sistema de señas a su corrección
# =============================================================================
DICCIONARIO_ERRORES = {
    # Errores comunes de letras similares
    "recepsiona": "recibido",
    "grasis": "gracias",
    "ola": "hola",
    "buenas noches": "buenas noches",
    "komo": "como",
    "kien": "quien",
    "ke": "que",
    "kiero": "quiero",
    "porfavor": "por favor",
    "porfa": "por favor",
    "xq": "porque",
    "tbn": "también",
    "tmbn": "también",
    "tb": "también",
    "qiero": "quiero",
    "decir": "decir",
    "aber": "haber",
    "aver": "a ver",
    "asia": "hacia",
    "habia": "había",
    "esta": "está",
    "tambien": "también",
    "senal": "señal",
    "lengaje": "lenguaje",
    "lenguage": "lenguaje",
    "comunicasion": "comunicación",
    "comunicacion": "comunicación",
    # Palabras del contexto de señas
    "s i": "sí",
    "n o": "no",
    "h o l a": "hola",
    "g r a c i a s": "gracias",
}

# =============================================================================
# REGLAS DE CORRECCIÓN GRAMATICAL
# Pares de (patrón_regex, reemplazo) para correcciones automáticas
# =============================================================================
REGLAS_GRAMATICA = [
    # Duplicar letras accidentales: "aabbcc" -> "abc"
    (r'(.)\1{2,}', r'\1'),  # Más de 2 letras iguales consecutivas
    
    # Espacios múltiples -> un espacio
    (r'\s+', ' '),
    
    # Corregir "i" sola como vocal -> "y" al final
    # (contexto de señas: la I se puede confundir)
    (r'\bi\b(?!\s*$)', 'y'),
    
    # Capitalizar primera letra de la frase
    # (se aplica después)
    
    # Quitar espacios al inicio y final
    (r'^\s+|\s+$', ''),
]


class IaCorrector:
    """
    Clase de corrección inteligente híbrida.
    Primero intenta corrección local, luego API externa si está disponible.
    """

    def __init__(self, api_key: Optional[str] = None, usar_api: bool = False):
        """
        Inicializa el corrector con configuración opcional de API.
        
        Args:
            api_key: Clave de API para servicio externo (opcional)
            usar_api: Si se debe usar corrección con IA externa
        """
        # Configuración de API externa
        self.api_key = api_key
        self.usar_api = usar_api and (api_key is not None)
        
        # Hilo actual de corrección (para no bloquear la UI)
        self.hilo_corrector: Optional[threading.Thread] = None
        
        # Umbral de longitud para activar IA externa
        # Si la frase tiene más palabras que esto, considerar IA
        self.umbral_longitud_api = 5
        
        # Resultado de la última corrección asíncrona
        self.ultimo_resultado: Optional[str] = None
        
        print("[IaCorrector] Sistema de corrección inicializado")
        if self.usar_api:
            print("[IaCorrector] Modo: Híbrido (local + API externa)")
        else:
            print("[IaCorrector] Modo: Solo corrección local")

    def corregir_local(self, frase: str) -> str:
        """
        Aplica corrección local usando diccionario y reglas gramaticales.
        Es rápida, funciona offline, sin latencia.
        
        Args:
            frase: Frase a corregir (puede tener errores)
            
        Returns:
            Frase corregida localmente
        """
        if not frase or not frase.strip():
            return frase
        
        # Paso 1: Convertir a minúsculas para comparación uniforme
        texto = frase.lower().strip()
        
        # Paso 2: Aplicar diccionario de errores comunes
        # Buscar frases completas del diccionario primero
        for error, correcto in DICCIONARIO_ERRORES.items():
            if error in texto:
                texto = texto.replace(error, correcto)
        
        # Paso 3: Aplicar reglas gramaticales con regex
        for patron, reemplazo in REGLAS_GRAMATICA:
            texto = re.sub(patron, reemplazo, texto)
        
        # Paso 4: Corregir palabras individuales
        palabras = texto.split()
        palabras_corregidas = []
        
        for palabra in palabras:
            # Buscar la palabra en el diccionario de errores
            if palabra in DICCIONARIO_ERRORES:
                palabras_corregidas.append(DICCIONARIO_ERRORES[palabra])
            else:
                palabras_corregidas.append(palabra)
        
        texto = ' '.join(palabras_corregidas)
        
        # Paso 5: Capitalizar la primera letra de la frase
        if texto:
            texto = texto[0].upper() + texto[1:]
        
        # Paso 6: Agregar punto final si no tiene puntuación
        if texto and texto[-1] not in '.!?,':
            texto += '.'
        
        return texto

    def debe_usar_api(self, frase: str, confianza_promedio: float = 1.0) -> bool:
        """
        Decide si debe usar la API externa para corrección.
        Solo se activa cuando vale la pena el costo/latencia.
        
        Args:
            frase: Frase a evaluar
            confianza_promedio: Confianza promedio del reconocimiento (0-1)
            
        Returns:
            True si se debe usar la API
        """
        # No usar API si no está configurada
        if not self.usar_api:
            return False
        
        # Contar palabras en la frase
        num_palabras = len(frase.split())
        
        # Usar API si la frase es larga (más de umbral palabras)
        if num_palabras >= self.umbral_longitud_api:
            return True
        
        # Usar API si la confianza promedio fue baja
        if confianza_promedio < 0.5:
            return True
        
        # Detectar palabras problemáticas (posibles errores del sistema de señas)
        palabras_sospechosas = 0
        for palabra in frase.lower().split():
            # Una palabra de 1-2 caracteres suele ser un error
            if len(palabra) <= 2 and palabra not in ['sí', 'no', 'yo', 'él', 'un', 'en', 'de']:
                palabras_sospechosas += 1
        
        # Si más del 30% son sospechosas, usar API
        if num_palabras > 0 and palabras_sospechosas / num_palabras > 0.3:
            return True
        
        return False

    def _construir_prompt(self, frase: str) -> str:
        """
        Construye el prompt para enviar a la IA externa.
        
        Args:
            frase: Frase a corregir
            
        Returns:
            Prompt completo para la API
        """
        return (
            "Corrige la siguiente frase proveniente de un sistema de reconocimiento "
            "de lenguaje de señas. Puede contener errores. Mantén el significado "
            "original, mejora gramática y hazla natural. No agregues información nueva. "
            f"Frase: {frase}"
        )

    def _llamar_api_anthropic(self, frase: str) -> Optional[str]:
        """
        Llama a la API de Anthropic (Claude) para corrección avanzada.
        
        Args:
            frase: Frase a corregir
            
        Returns:
            Frase corregida por la IA o None si falla
        """
        try:
            import anthropic  # Importar aquí para no fallar si no está instalado
            
            # Crear cliente con la API key configurada
            client = anthropic.Anthropic(api_key=self.api_key)
            
            # Construir el prompt
            prompt = self._construir_prompt(frase)
            
            # Llamar a la API con el modelo más ligero disponible
            mensaje = client.messages.create(
                model="claude-haiku-4-5-20251001",  # Modelo más rápido y económico
                max_tokens=200,                      # Limitar tokens de respuesta
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extraer el texto de la respuesta
            if mensaje.content and len(mensaje.content) > 0:
                return mensaje.content[0].text.strip()
            
            return None
            
        except ImportError:
            print("[IaCorrector] anthropic no instalado. Usando corrección local.")
            return None
        except Exception as e:
            print(f"[IaCorrector] Error API Anthropic: {e}")
            return None

    def _llamar_api_openai(self, frase: str) -> Optional[str]:
        """
        Llama a la API de OpenAI para corrección avanzada (alternativa).
        
        Args:
            frase: Frase a corregir
            
        Returns:
            Frase corregida por la IA o None si falla
        """
        try:
            import openai  # Importar aquí para no fallar si no está instalado
            
            # Crear cliente con la API key
            client = openai.OpenAI(api_key=self.api_key)
            
            # Construir el prompt
            prompt = self._construir_prompt(frase)
            
            # Llamar a la API con modelo económico
            respuesta = client.chat.completions.create(
                model="gpt-3.5-turbo",   # Modelo más económico de OpenAI
                max_tokens=200,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extraer el texto de la respuesta
            return respuesta.choices[0].message.content.strip()
            
        except ImportError:
            print("[IaCorrector] openai no instalado. Usando corrección local.")
            return None
        except Exception as e:
            print(f"[IaCorrector] Error API OpenAI: {e}")
            return None

    def corregir(
        self,
        frase: str,
        confianza: float = 1.0,
        callback: Optional[Callable[[str], None]] = None
    ) -> str:
        """
        Corrige una frase usando la estrategia apropiada.
        Si se necesita API, la llamada es asíncrona (no bloquea).
        
        Args:
            frase: Frase a corregir
            confianza: Confianza promedio del reconocimiento
            callback: Función a llamar cuando la corrección IA termine
            
        Returns:
            Frase corregida localmente (la corrección IA se entrega vía callback)
        """
        # Primero aplicar corrección local siempre
        frase_local = self.corregir_local(frase)
        
        print(f"[IaCorrector] Original: '{frase}'")
        print(f"[IaCorrector] Corrección local: '{frase_local}'")
        
        # Decidir si también usar API
        if self.debe_usar_api(frase, confianza) and callback is not None:
            # Ejecutar corrección API en hilo separado para no bloquear la UI
            self.hilo_corrector = threading.Thread(
                target=self._corregir_con_api_async,
                args=(frase, callback),
                daemon=True  # El hilo se cierra cuando la app se cierra
            )
            self.hilo_corrector.start()
            print("[IaCorrector] Corrección IA iniciada en segundo plano...")
        
        # Retornar inmediatamente con la corrección local
        return frase_local

    def _corregir_con_api_async(
        self,
        frase: str,
        callback: Callable[[str], None]
    ) -> None:
        """
        Función interna que ejecuta la corrección API de forma asíncrona.
        Se ejecuta en un hilo separado.
        
        Args:
            frase: Frase a corregir
            callback: Función a llamar con el resultado
        """
        resultado = None
        
        # Intentar con Anthropic primero
        resultado = self._llamar_api_anthropic(frase)
        
        # Si falló Anthropic, intentar con OpenAI
        if resultado is None:
            resultado = self._llamar_api_openai(frase)
        
        # Si ambas fallaron, usar la corrección local como fallback
        if resultado is None:
            resultado = self.corregir_local(frase)
            print("[IaCorrector] API falló, usando corrección local como fallback")
        else:
            print(f"[IaCorrector] Corrección IA completada: '{resultado}'")
        
        # Guardar el resultado
        self.ultimo_resultado = resultado
        
        # Llamar al callback con el resultado final
        callback(resultado)

    def corregir_sincrono(self, frase: str) -> str:
        """
        Versión síncrona de corrección (bloquea hasta terminar).
        Solo usar fuera de la interfaz gráfica (ej: exportación).
        
        Args:
            frase: Frase a corregir
            
        Returns:
            Frase completamente corregida
        """
        # Aplicar corrección local
        resultado = self.corregir_local(frase)
        
        # Si se puede usar API, intentar corrección completa
        if self.debe_usar_api(frase):
            api_resultado = self._llamar_api_anthropic(frase)
            if api_resultado:
                resultado = api_resultado
        
        return resultado
