"""worldChef.py

Unifica las utilidades del proyecto en un único menú:
- Normalizador
- Patrones con RE
- Resumen simple
- Extracción NER 
- Palabras clave 
- Sentimiento 

El script intenta manejar faltas de dependencias de forma informativa.
"""

import re
import sys
from collections import Counter

# Intentos de import de dependencias; se comprueba en tiempo de ejecución
try:
	import spacy
except Exception:
	spacy = None

try:
	import nltk
	from nltk.corpus import stopwords
	from nltk.tokenize import word_tokenize
except Exception:
	nltk = None

# Transformers (pipeline) — import al inicio para evitar imports dinámicos dentro de funciones
try:
    from transformers import pipeline
except Exception:
    pipeline = None

def cargar_modelo_spacy():
	"""Intenta cargar un modelo de spaCy en español.
	Devuelve un objeto nlp o None si no está disponible.
	"""
	if spacy is None:
		print("Aviso: spaCy no está instalado. Algunas funciones no estarán disponibles.")
		return None

	modelos = ["es_core_news_sm", "es_core_news_md", "xx_sent_ud_sm"]
	for m in modelos:
		try:
			nlp = spacy.load(m)
			if "sentencizer" not in nlp.pipe_names:
				try:
					nlp.add_pipe("sentencizer")
				except Exception:
					pass
			return nlp
		except Exception:
			continue

	# Fallback a modelo vacío (no etiquetado POS/NER)
	try:
		nlp = spacy.blank("es")
		nlp.add_pipe("sentencizer")
		return nlp
	except Exception:
		return None


# ----------------------
#  Normalizador
# ----------------------
CORRECCIONES_COMUNES = {
	"haiga": "haya",
	"naiden": "nadie",
	"nadien": "nadie",
	"aserca": "acerca",
	"enserio": "en serio",
	"haber": "a ver",
	"iva": "iba",
}

SUSTANTIVOS_NO_NEUTROS = {
	"casa": "la casa",
	"persona": "la persona",
	"gente": "la gente",
	"niño": "el niño",
	"niña": "la niña",
	"camisa": "la camisa"
}

def corregir_palabras(doc):
	if doc is None:
		return ""  # no podemos corregir sin nlp

	corregido = []
	for i, token in enumerate(doc):
		palabra = token.text.lower()
		if palabra in CORRECCIONES_COMUNES:
			corregido.append(CORRECCIONES_COMUNES[palabra])
			continue

		if token.text.lower() == "lo" and token.pos_ == "DET":
			if i + 1 < len(doc):
				siguiente = doc[i + 1].lemma_.lower()
				if siguiente in SUSTANTIVOS_NO_NEUTROS:
					corregido.append(SUSTANTIVOS_NO_NEUTROS[siguiente])
					continue
			corregido.append("el")
			continue

		if palabra == "haber":
			if i > 0 and doc[i - 1].text.lower() in ["vamos", "voy", "van", "vas", "quiera"]:
				corregido.append("a ver")
				continue

		if i > 0 and palabra == doc[i - 1].text.lower():
			continue

		corregido.append(token.text)

	return " ".join(corregido)


def normalizador_texto(texto, nlp):
	if not texto or len(texto.strip()) == 0:
		return None

	if nlp is None:
		# Fallback simple
		palabras = texto.split()
		sin_repeticiones = []
		for i, w in enumerate(palabras):
			if i == 0 or w.lower() != palabras[i - 1].lower():
				sin_repeticiones.append(w)
		return {
			"original": texto,
			"lematizado": "(se necesita spaCy para lematizar)",
			"sin_repeticiones": " ".join(sin_repeticiones),
			"corregido": "(se necesita spaCy para correcciones avanzadas)"
		}

	doc = nlp(texto)
	lematizado = " ".join([token.lemma_ for token in doc])

	palabras = texto.split()
	sin_repeticiones = []
	for i, w in enumerate(palabras):
		if i == 0 or w.lower() != palabras[i - 1].lower():
			sin_repeticiones.append(w)
	sin_repeticiones = " ".join(sin_repeticiones)

	texto_corregido = corregir_palabras(doc)

	return {
		"original": texto,
		"lematizado": lematizado,
		"sin_repeticiones": sin_repeticiones,
		"corregido": texto_corregido
	}


# ----------------------
#  Patrones con RE
# ----------------------
patron_fechas = r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b"
patron_dinero = r"\b(?:€?\s?\d{1,3}(?:[\.,]\d{3})*(?:[\.,]\d+)?\s?(?:€|euros|USD|\$)|\$\d+(?:\.\d+)?\b)"
patron_email = r"\b[\w\.-]+@[\w\.-]+\.\w{2,4}\b"

def encontrar_fechas(texto):
	return re.findall(patron_fechas, texto)

def encontrar_dinero(texto):
	return re.findall(patron_dinero, texto)

def encontrar_correos(texto):
	return re.findall(patron_email, texto)


# ----------------------
#  Resumen simple
# ----------------------
def resumen_simple(texto, n=3, nlp=None):
	if not texto or not texto.strip():
		return "Error: texto vacío."

	if nlp is None:
		# Fallback: dividir por puntos
		oraciones = [s.strip() for s in texto.split('.') if s.strip()]
	else:
		doc = nlp(texto)
		oraciones = list(doc.sents)

	if len(oraciones) <= n:
		return texto

	puntuaciones = []
	for i, oracion in enumerate(oraciones):
		puntaje = 0
		tokens = oracion if nlp else re.findall(r"\w+", oracion)
		if nlp:
			sustantivos = [t for t in oracion if t.pos_ == "NOUN"]
			puntaje += len(sustantivos)
			longitud = len(oracion.text)
		else:
			# conteo aproximado
			sustantivos = [w for w in tokens if len(w) > 2]
			puntaje += len(sustantivos)
			longitud = len(oracion)

		puntaje -= longitud / 200.0
		if i == 0:
			puntaje += 1
		puntuaciones.append((i, puntaje))

	mejores = sorted(puntuaciones, key=lambda x: x[1], reverse=True)[:n]
	indices = sorted([idx for idx, _ in mejores])
	if nlp:
		resumen = " ".join(oraciones[i].text.strip() for i in indices)
	else:
		resumen = " ".join(oraciones[i] for i in indices)

	return resumen


# ----------------------
#  Extracción NER 
# ----------------------
def extraer_entidades(texto, nlp):
	if nlp is None:
		print("Aviso: spaCy no disponible — NER no puede ejecutarse.")
		return {}
	doc = nlp(texto)
	def extraer(tipo):
		return [ent.text for ent in doc.ents if ent.label_ == tipo]

	return {
		'Personas': set(extraer('PER')),
		'Lugares': set(extraer('LOC')),
		'Empresas': set(extraer('ORG')),
		'Fechas': set(extraer('DATE')),
		'Cantidades': set([ent.text for ent in doc.ents if ent.label_ == 'QUANTITY'])
	}


# ----------------------
#  Palabras clave - M
# ----------------------
def extraer_palabras_clave(texto, nlp=None):
	if not texto or not texto.strip():
		return None

	if nltk is not None:
		try:
			nltk.data.find('tokenizers/punkt')
			nltk.data.find('corpora/stopwords')
		except Exception:
			print("Descargando recursos NLTK necesarios (punkt, stopwords)...")
			try:
				nltk.download('punkt')
				nltk.download('stopwords')
			except Exception:
				pass

	try:
		tokens = word_tokenize(texto.lower())

		palabras_parada = set(stopwords.words('spanish'))

		tokens_filtrados = [t for t in tokens if t.isalnum() and t not in palabras_parada and len(t) > 2]
        
		top_5 = Counter(tokens_filtrados).most_common(5)
	except Exception:
		top_5 = []

	sustantivos_relevantes = []
	verbos_principales = []
	if nlp is not None:
		try:
			doc = nlp(texto)
			sustantivos = [token.text for token in doc if token.pos_ == 'NOUN']

			verbos = [token.text for token in doc if token.pos_ == 'VERB']

			sustantivos_relevantes = Counter(sustantivos).most_common(5)

			verbos_principales = Counter(verbos).most_common(5)
		except Exception:
			pass

	return {
		'top_5_palabras': top_5,
		'sustantivos': sustantivos_relevantes,
		'verbos': verbos_principales
	}

# ----------------------
#  Análisis de sentimiento
# ----------------------

def sentimiento_es(texto):
	if not texto or not texto.strip():
		return "Error: texto vacío.", 0.0, ""

	# Usar el pipeline importado globalmente (si está disponible)
	if pipeline is None:
		return "Error: transformers no instalado.", 0.0, "transformers missing"

	try:
		clasificador = pipeline(
			"sentiment-analysis",
			model="nlptown/bert-base-multilingual-uncased-sentiment"
		)
		resultado = clasificador(texto)[0]
		etiqueta = resultado.get('label', '')
		puntuacion = resultado.get('score', 0.0)
		if "1" in etiqueta or "2" in etiqueta:
			sentimiento = "Negativo"
		elif "3" in etiqueta:
			sentimiento = "Neutral"
		else:
			sentimiento = "Positivo"
		return sentimiento, puntuacion, etiqueta
	except Exception as e:
		return "Error", 0.0, str(e)


def menu_principal():
	nlp = cargar_modelo_spacy()

	while True:
		print("\n=== worldChef: Menú de utilidades ===")
		print("1) Normalizador de texto")
		print("2) Buscar patrones (fechas, dinero, correos)")
		print("3) Resumen simple")
		print("4) Extracción de entidades")
		print("5) Palabras clave")
		print("6) Análisis de sentimiento")
		print("0) Salir")

		opcion = input("Selecciona una opción: ").strip()

		if opcion == '0':
			print("Saliendo.")
			break

		if opcion == '1':
			texto = input("Ingresa texto a normalizar:\n> ")
			res = normalizador_texto(texto, nlp)
			if res is None:
				print("Texto inválido.")
			else:
				print("\n--- RESULTADOS ---")
				print("Texto Original:\n", res['original'])
				print("\nTexto Lematizado:\n", res['lematizado'])
				print("\nTexto sin Repeticiones:\n", res['sin_repeticiones'])
				print("\nTexto Corregido:\n", res['corregido'])

		elif opcion == '2':
			texto = input("Introduce texto para analizar patrones:\n")
			print('\nFechas encontradas:', encontrar_fechas(texto) or 'Ninguna')
			print('Cifras de dinero:', encontrar_dinero(texto) or 'Ninguna')
			print('Correos electrónicos:', encontrar_correos(texto) or 'Ninguno')

		elif opcion == '3':
			texto = input("Introduce texto para resumir:\n")
			resumen = resumen_simple(texto, n=3, nlp=nlp)
			print('\n--- RESUMEN ---')
			print(resumen)

		elif opcion == '4':
			texto = input("Introduce texto para extraer entidades (NER):\n")
			entidades = extraer_entidades(texto, nlp)
			for k, v in entidades.items():
				print(f"{k}: {v if v else 'Ninguno detectado'}")

		elif opcion == '5':
			texto = input("Introduce texto para extraer palabras clave:\n")
			resultado = extraer_palabras_clave(texto, nlp=nlp)
			if not resultado:
				print("No se pudieron extraer palabras clave.")
			else:
				print('\nTop 5 palabras:', resultado['top_5_palabras'])
				print('Sustantivos relevantes:', resultado['sustantivos'])
				print('Verbos principales:', resultado['verbos'])

		elif opcion == '6':
			texto = input("Introduce texto para análisis de sentimiento:\n")
			sentimiento, score, raw = sentimiento_es(texto)
			if sentimiento == "Error":
				print(f"Ocurrió un error: {raw}")
			else:
				print(f"Resultado: {sentimiento} (Confianza: {score:.4f}) Estrellas: {raw}")

		else:
			print("Opción no válida, intenta de nuevo.")


if __name__ == '__main__':
	try:
		menu_principal()
	except KeyboardInterrupt:
		print("\nInterrumpido por el usuario.\n")
