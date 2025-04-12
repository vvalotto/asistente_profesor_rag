# app/utils/text_normalizer.py

import re
import logging
import unicodedata
from pathlib import Path
from typing import List, Dict, Optional

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Descargar recursos de NLTK necesarios
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

logger = logging.getLogger(__name__)


class TextNormalizer:
    """
    Normaliza y estructura textos extraídos para su procesamiento posterior.
    """

    def __init__(self, language: str = 'spanish'):
        """
        Inicializa el normalizador.

        Args:
            language: Idioma para procesamiento (default: español)
        """
        self.language = language
        self.stopwords = set(stopwords.words(language)) if language in stopwords._fileids else set()

    def normalize_text(self, text: str) -> str:
        """
        Normaliza un texto aplicando varias transformaciones.

        Args:
            text: Texto a normalizar

        Returns:
            Texto normalizado
        """
        # Eliminar marcadores de página
        text = re.sub(r'---\s*[pP][aá]gina\s+\d+\s*---', ' ', text)

        # Eliminar números de página aislados
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)

        # Convertir a minúsculas
        text = text.lower()

        # Normalizar caracteres Unicode (acentos, etc.)
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')

        # Corregir términos específicos conocidos
        text = text.replace("licitacion", "elicitacion")

        # Eliminar caracteres especiales pero mantener puntuación importante
        text = re.sub(r'[^\w\s\.\,\;\:\-\(\)\[\]\{\}\"\'\¿\?\¡\!]', ' ', text)

        # Eliminar espacios múltiples
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def segment_into_paragraphs(self, text: str) -> List[str]:
        """
        Segmenta el texto en párrafos de manera más efectiva.

        Args:
            text: Texto a segmentar

        Returns:
            Lista de párrafos
        """
        # Pre-procesamiento para facilitar la detección de párrafos
        # Asegurar que hay saltos de línea después de los puntos seguidos de mayúsculas
        text = re.sub(r'([.!?])\s+([A-Z])', r'\1\n\2', text)

        # Primero, dividir por saltos de línea múltiples
        paragraphs = re.split(r'\n\s*\n', text)

        # Si no se detectaron suficientes párrafos, usar estrategia alternativa
        if len(paragraphs) <= 3:
            # Dividir por puntos seguidos de una letra mayúscula
            paragraphs = []
            current_paragraph = ""

            sentences = sent_tokenize(text, language='spanish')
            for sentence in sentences:
                if len(current_paragraph) > 0 and any(kw in sentence.lower() for kw in
                                                      ['el ciclo', 'la siguiente fase', 'sub-procesos', 'luego,',
                                                       'en paralelo']):
                    paragraphs.append(current_paragraph.strip())
                    current_paragraph = sentence
                else:
                    if current_paragraph:
                        current_paragraph += " " + sentence
                    else:
                        current_paragraph = sentence

            if current_paragraph:
                paragraphs.append(current_paragraph.strip())

        # Filtrar párrafos vacíos y normalizar espacios
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        # Filtrar párrafos que solo contienen números (posibles números de página)
        paragraphs = [p for p in paragraphs if not re.match(r'^\d+$', p.strip())]

        # Asegurar que hay contenido de párrafos
        if not paragraphs:
            # Si no hay párrafos, mantener el texto original como un solo párrafo
            paragraphs = [text.strip()]

        return paragraphs

    def process_file(self, input_path: str, output_path: Optional[str] = None) -> Dict:
        """
        Procesa un archivo de texto, normalizándolo y estructurándolo.

        Args:
            input_path: Ruta al archivo de texto
            output_path: Ruta para guardar el resultado (opcional)

        Returns:
            Estadísticas del procesamiento
        """
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo: {input_path}")

        # Determinar ruta de salida si no se proporciona
        if output_path is None:
            output_path = input_path.with_suffix('.norm.txt')
        else:
            output_path = Path(output_path)

        # Leer contenido
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Normalizar texto
        normalized_text = self.normalize_text(content)

        # Segmentar en párrafos con la función mejorada
        paragraphs = self.segment_into_paragraphs(normalized_text)

        # Extraer estadísticas para metadatos
        sentences = []
        for para in paragraphs:
            # Especificar el idioma español para la tokenización
            para_sentences = sent_tokenize(para, language='spanish')
            sentences.extend(para_sentences)

        # También especificar el idioma español para la tokenización de palabras
        words = word_tokenize(" ".join(paragraphs), language='spanish')
        words_no_stopwords = [w for w in words if w.isalnum() and w not in self.stopwords]

        # Escribir resultado con formato apropiado para párrafos
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n\n".join(paragraphs))

        # Compilar estadísticas
        stats = {
            "input_file": str(input_path),
            "output_file": str(output_path),
            "paragraphs": len(paragraphs),
            "sentences": len(sentences),
            "words": len(words),
            "unique_words": len(set(words_no_stopwords))
        }

        logger.info(f"Procesado {input_path.name}: {stats['paragraphs']} párrafos, {stats['sentences']} oraciones")

        return stats