"""
Extracción de Palabras Clave usando spaCy y NLTK

Este programa extrae palabras clave importantes de un texto en español:
- Top 5 palabras más frecuentes (NLTK): Palabras significativas más comunes
- Sustantivos relevantes (spaCy): Nombres y entidades principales
- Verbos principales (spaCy): Acciones principales del texto

Requiere:
- spacy: Librería de procesamiento de lenguaje natural
- nltk: Kit de herramientas de lenguaje natural
- Modelo de spaCy en español: es_core_news_sm
"""

import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

# ============================================
# DESCARGA DE DATOS NECESARIOS
# ============================================
# NLTK necesita descargar recursos para tokenizar texto (punkt_tab)
# y palabras vacías en español (stopwords)
try:
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Descargando recursos de NLTK...")
    nltk.download('punkt_tab')
    nltk.download('stopwords')

# ============================================
# CARGA DEL MODELO DE SPACY EN ESPAÑOL
# ============================================
# spaCy necesita un modelo previamente entrenado (es_core_news_sm)
# Este modelo permite identificar verbos, sustantivos y otras características
try:
    nlp = spacy.load('es_core_news_sm')
except OSError:
    print("Error: Modelo de spaCy no encontrado.")
    print("Por favor, ejecuta: python -m spacy download es_core_news_sm")
    exit()


def extraer_palabras_clave(texto):
    """
    Extrae palabras clave importantes del texto en español.
    
    Argumentos:
        texto (str): Texto del que extraer palabras clave
        
    Retorna:
        dict: Diccionario con tres listas de palabras clave o None si hay error
    """
    
    # ============================================
    # VALIDACIÓN DEL TEXTO DE ENTRADA
    # ============================================
    # Verificamos que el texto no esté vacío antes de procesarlo
    if not texto or len(texto.strip()) == 0:
        print("Error: El texto está vacío. Por favor proporciona un texto válido.")
        return None
    
    try:
        # ============================================
        # PROCESAMIENTO DEL TEXTO CON SPACY
        # ============================================
        # spaCy analiza el texto y etiqueta cada palabra con su tipo
        # (sustantivo, verbo, adjetivo, etc.)
        doc = nlp(texto)
        
        # ============================================
        # 1. TOP 5 PALABRAS MÁS FRECUENTES (NLTK)
        # ============================================
        # Paso 1: Tokenizar el texto (dividir en palabras individuales)
        # .lower() convierte todo a minúsculas para consistencia
        tokens = word_tokenize(texto.lower())
        
        # Paso 2: Obtener palabras vacías en español
        # Palabras vacías = palabras comunes sin significado (el, la, de, que, etc.)
        palabras_parada = set(stopwords.words('spanish'))
        
        # Paso 3: Filtrar tokens
        # Guardamos solo palabras que:
        # - Son alfanuméricos (sin signos especiales)
        # - NO están en la lista de palabras vacías
        # - Tienen más de 2 caracteres (elimina caracteres sueltos)
        tokens_filtrados = [
            token for token in tokens 
            if token.isalnum() and token not in palabras_parada and len(token) > 2
        ]
        
        # Paso 4: Contar frecuencias
        # Counter cuenta cuántas veces aparece cada palabra
        # .most_common(5) retorna las 5 palabras más frecuentes
        top_5_palabras = Counter(tokens_filtrados).most_common(5)
        
        # ============================================
        # 2. SUSTANTIVOS RELEVANTES (SPACY)
        # ============================================
        # Recorremos cada token procesado por spaCy
        # token.pos_ nos dice la parte del discurso (NOUN, VERB, ADJ, etc.)
        # NOUN = sustantivo (nombres de personas, lugares, cosas)
        sustantivos = [token.text for token in doc if token.pos_ == 'NOUN']
        
        # Contar frecuencias de sustantivos y obtener los 5 más comunes
        sustantivos_relevantes = Counter(sustantivos).most_common(5)
        
        # ============================================
        # 3. VERBOS PRINCIPALES (SPACY)
        # ============================================
        # VERB = verbo (acciones, estados)
        # Extraemos todos los verbos del texto
        verbos = [token.text for token in doc if token.pos_ == 'VERB']
        
        # Contar frecuencias de verbos y obtener los 5 más comunes
        verbos_principales = Counter(verbos).most_common(5)
        
        # ============================================
        # RETORNO DE RESULTADOS
        # ============================================
        # Devolvemos un diccionario con los tres tipos de palabras clave extraídas
        return {
            'top_5_palabras': top_5_palabras,
            'sustantivos': sustantivos_relevantes,
            'verbos': verbos_principales
        }
    
    except Exception as e:
        # Capturamos cualquier error inesperado y lo reportamos
        print(f"Error al procesar el texto: {str(e)}")
        return None


def mostrar_resultados(palabras_clave):
    """
    Muestra los resultados de forma clara y organizada en consola.
    
    Argumentos:
        palabras_clave (dict): Diccionario con los resultados de la extracción
    """
    
    # Validar que el diccionario tiene los datos esperados
    if not palabras_clave:
        print("Error: No hay resultados para mostrar.")
        return
    
    # ============================================
    # ENCABEZADO DEL RESULTADO
    # ============================================
    print("\n" + "="*50)
    print("RESULTADOS DE EXTRACCIÓN DE PALABRAS CLAVE")
    print("="*50)
    
    # ============================================
    # MOSTRAR TOP 5 PALABRAS MÁS FRECUENTES
    # ============================================
    print("\nTOP 5 PALABRAS MÁS FRECUENTES:")
    # Iteramos sobre la lista de palabras con sus frecuencias
    # enumerate(lista, 1) numera desde 1
    # palabra es el texto, freq es la cantidad de veces que aparece
    for i, (palabra, freq) in enumerate(palabras_clave['top_5_palabras'], 1):
        print(f"  {i}. {palabra}: {freq} veces")
    
    # ============================================
    # MOSTRAR SUSTANTIVOS RELEVANTES
    # ============================================
    print("\nSUSTANTIVOS RELEVANTES:")
    # Mismo proceso: mostramos cada sustantivo y su frecuencia
    for i, (sustantivo, freq) in enumerate(palabras_clave['sustantivos'], 1):
        print(f"  {i}. {sustantivo}: {freq} veces")
    
    # ============================================
    # MOSTRAR VERBOS PRINCIPALES
    # ============================================
    print("\nVERBOS PRINCIPALES:")
    # Mismo proceso: mostramos cada verbo y su frecuencia
    for i, (verbo, freq) in enumerate(palabras_clave['verbos'], 1):
        print(f"  {i}. {verbo}: {freq} veces")
    
    # Línea de cierre
    print("\n" + "="*50 + "\n")


# ============================================
# PUNTO DE ENTRADA DEL PROGRAMA
# ============================================
# Este bloque solo se ejecuta si el archivo se ejecuta directamente
# (no si se importa como módulo en otro archivo)
if __name__ == "__main__":
    try:
        # ============================================
        # TEXTO DE EJEMPLO
        # ============================================
        # Texto en español sobre procesamiento de lenguaje natural
        texto = """
        El procesamiento del lenguaje natural es un campo fascinante de la inteligencia artificial.
        Los modelos de aprendizaje automático pueden procesar y entender el lenguaje humano.
        Las técnicas de aprendizaje profundo han revolucionado la forma en que procesamos datos de texto.
        Las bibliotecas de Python como spaCy y NLTK proporcionan una excelente funcionalidad.
        """
        
        # ============================================
        # MOSTRAR TEXTO DE ENTRADA
        # ============================================
        print("TEXTO DE ENTRADA:")
        print("-" * 50)
        print(texto.strip())
        
        # ============================================
        # EXTRAER PALABRAS CLAVE
        # ============================================
        # Llamamos a la función principal que extrae las palabras clave
        resultado = extraer_palabras_clave(texto)
        
        # ============================================
        # MOSTRAR RESULTADOS
        # ============================================
        # Si la extracción fue exitosa, mostramos los resultados
        # Si hubo error, resultado será None y no mostraremos nada
        if resultado:
            mostrar_resultados(resultado)
        else:
            print("No se pudieron extraer las palabras clave.")
    
    except KeyboardInterrupt:
        # Si el usuario presiona Ctrl+C, salimos gracefully
        print("\n\nPrograma interrumpido por el usuario.")
    except Exception as e:
        # Capturamos cualquier error no previsto
        print(f"Error inesperado: {str(e)}")
        print("Por favor, verifica que el texto es válido e intenta de nuevo.")
