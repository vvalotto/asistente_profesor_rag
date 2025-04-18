# app/ingestion/process_documents.py

import os
import argparse
import logging
import json
from pathlib import Path

from app.ingestion.document_chunker import DocumentChunker

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_documents(input_dir: str, output_dir: str,
                      chunk_size: int = 1024, chunk_overlap: int = 200,
                      file_pattern: str = "*.txt",
                      metadata_file: str = None):
    """
    Procesa documentos con el DocumentChunker.

    Args:
        input_dir: Directorio con archivos a procesar
        output_dir: Directorio para guardar fragmentos
        chunk_size: Tamaño máximo de fragmento en tokens
        chunk_overlap: Cantidad de tokens que se solapan entre fragmentos
        file_pattern: Patrón glob para filtrar archivos
        metadata_file: Archivo JSON opcional con metadatos por archivo
    """
    logger.info(f"Iniciando procesamiento de documentos de {input_dir}")
    logger.info(f"Configuración: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")

    # Cargar metadatos si se proporciona un archivo
    file_metadata = {}
    if metadata_file:
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                file_metadata = json.load(f)
            logger.info(f"Metadatos cargados desde {metadata_file} para {len(file_metadata)} archivos")
        except Exception as e:
            logger.error(f"Error al cargar metadatos: {str(e)}")

    # Inicializar chunker
    chunker = DocumentChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # Procesar directorio
    if os.path.isdir(input_dir):
        stats = chunker.process_directory(
            input_dir=input_dir,
            output_dir=output_dir,
            file_pattern=file_pattern,
            metadata=None  # Usaremos metadatos por archivo
        )

        logger.info(f"Procesamiento completo: {stats['processed_files']} archivos, "
                    f"{stats['total_chunks']} fragmentos, {stats['total_tokens']} tokens")

    # Procesar un solo archivo
    elif os.path.isfile(input_dir):
        file_path = input_dir

        # Obtener metadatos para este archivo si existen
        metadata = None
        if file_path in file_metadata:
            metadata = file_metadata[file_path]

        chunks, stats = chunker.process_file(
            file_path=file_path,
            output_dir=output_dir,
            metadata=metadata
        )

        # Evaluar calidad de los fragmentos
        quality_metrics = chunker.evaluate_chunks_quality(chunks)

        # Guardar métricas de calidad
        quality_file = Path(output_dir) / Path(file_path).stem / "quality_metrics.json"
        with open(quality_file, 'w', encoding='utf-8') as f:
            json.dump(quality_metrics, f, ensure_ascii=False, indent=2)

        logger.info(f"Archivo procesado: {Path(file_path).name} - {stats['total_chunks']} fragmentos")
        logger.info(f"Calidad: {quality_metrics['quality_score']}/100 - {quality_metrics['quality_interpretation']}")

    else:
        logger.error(f"La ruta de entrada no existe: {input_dir}")
        return


def main():
    parser = argparse.ArgumentParser(description="Procesa documentos para RAG, dividiéndolos en fragmentos óptimos")

    parser.add_argument("--input", required=True, help="Directorio o archivo de entrada")
    parser.add_argument("--output", required=True, help="Directorio de salida para los fragmentos")
    parser.add_argument("--chunk-size", type=int, default=1024, help="Tamaño máximo de fragmento en tokens")
    parser.add_argument("--chunk-overlap", type=int, default=200,
                        help="Cantidad de tokens que se solapan entre fragmentos")
    parser.add_argument("--file-pattern", default="*.txt",
                        help="Patrón para filtrar archivos (solo si input es directorio)")
    parser.add_argument("--metadata", help="Archivo JSON con metadatos por archivo")

    args = parser.parse_args()

    process_documents(
        input_dir=args.input,
        output_dir=args.output,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        file_pattern=args.file_pattern,
        metadata_file=args.metadata
    )


if __name__ == "__main__":
    main()