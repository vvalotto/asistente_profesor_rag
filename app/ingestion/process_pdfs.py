# app/ingestion/process_pdfs.py

import os
import logging
import argparse
import json
from pathlib import Path

from app.ingestion.pdf_extractor import PDFExtractor
from app.utils.text_normalizer import TextNormalizer

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_pdfs(raw_dir: str, processed_dir: str, normalized_dir: str):
    """
    Procesa PDFs: extrae texto y lo normaliza.

    Args:
        raw_dir: Directorio con PDFs originales
        processed_dir: Directorio para textos extraídos
        normalized_dir: Directorio para textos normalizados
    """
    # Crear directorios si no existen
    Path(normalized_dir).mkdir(exist_ok=True, parents=True)

    # Paso 1: Extraer texto de PDFs
    logger.info("Iniciando extracción de PDFs")
    extractor = PDFExtractor(raw_dir, processed_dir)
    extracted_files = extractor.extract_all_pdfs()

    # Paso 2: Normalizar textos extraídos
    logger.info("Iniciando normalización de textos")
    normalizer = TextNormalizer(language='spanish')

    processing_stats = []
    for text_file in extracted_files:
        output_path = Path(normalized_dir) / f"{Path(text_file).stem}.norm.txt"
        stats = normalizer.process_file(text_file, output_path)
        processing_stats.append(stats)

    # Guardar estadísticas de procesamiento
    stats_path = Path(normalized_dir) / "processing_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(processing_stats, f, indent=2)

    logger.info(f"Procesamiento completo. Estadísticas guardadas en {stats_path}")

    # Resumen final
    total_docs = len(processing_stats)
    total_paragraphs = sum(s["paragraphs"] for s in processing_stats)
    total_sentences = sum(s["sentences"] for s in processing_stats)

    logger.info(f"Resumen: {total_docs} documentos procesados, "
                f"{total_paragraphs} párrafos, {total_sentences} oraciones")


def main():
    parser = argparse.ArgumentParser(description="Procesa PDFs y normaliza textos")
    parser.add_argument("--raw", default="data/raw", help="Directorio con PDFs originales")
    parser.add_argument("--processed", default="data/processed/extracted",
                        help="Directorio para textos extraídos")
    parser.add_argument("--normalized", default="data/processed/normalized",
                        help="Directorio para textos normalizados")

    args = parser.parse_args()

    process_pdfs(args.raw, args.processed, args.normalized)


if __name__ == "__main__":
    main()