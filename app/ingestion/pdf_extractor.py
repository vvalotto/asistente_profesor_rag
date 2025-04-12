# app/ingestion/pdf_extractor.py

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional

import PyPDF2

# Alternativa: from pdfminer.high_level import extract_text

logger = logging.getLogger(__name__)


class PDFExtractor:
    """
    Extrae texto de archivos PDF y los guarda en formato procesado.
    """

    def __init__(self, raw_dir: str, processed_dir: str):
        """
        Inicializa el extractor con directorios de entrada/salida.

        Args:
            raw_dir: Directorio que contiene los PDFs originales
            processed_dir: Directorio donde se guardarán los textos extraídos
        """
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(exist_ok=True, parents=True)

    def extract_single_pdf(self, pdf_path: str, output_filename: Optional[str] = None) -> str:
        """
        Extrae texto de un único archivo PDF.

        Args:
            pdf_path: Ruta al archivo PDF
            output_filename: Nombre del archivo de salida (opcional)

        Returns:
            Ruta al archivo de texto extraído
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo: {pdf_path}")

        logger.info(f"Extrayendo texto de: {pdf_path}")

        # Nombre del archivo de salida
        if output_filename is None:
            output_filename = f"{pdf_path.stem}.txt"

        output_path = self.processed_dir / output_filename

        # Extracción con PyPDF2
        text_content = ""

        try:
            with open(pdf_path, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)

                # Extraer texto de cada página
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()

                    if page_text.strip():  # Si hay texto en la página
                        text_content += f"\n--- Página {page_num + 1} ---\n"
                        text_content += page_text

            # Guardar el texto extraído
            with open(output_path, 'w', encoding='utf-8') as text_file:
                text_file.write(text_content)

            logger.info(f"Texto extraído guardado en: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Error al procesar {pdf_path}: {str(e)}")
            raise

    def extract_all_pdfs(self) -> List[str]:
        """
        Extrae texto de todos los PDFs en el directorio raw.

        Returns:
            Lista de rutas a los archivos de texto extraídos
        """
        output_files = []

        for pdf_file in self.raw_dir.glob("*.pdf"):
            try:
                output_path = self.extract_single_pdf(pdf_file)
                output_files.append(output_path)
            except Exception as e:
                logger.error(f"Error procesando {pdf_file}: {str(e)}")
                continue

        logger.info(f"Procesados {len(output_files)} archivos PDF")
        return output_files

    def get_extraction_metadata(self) -> Dict:
        """
        Devuelve metadatos sobre los archivos procesados.

        Returns:
            Diccionario con metadatos de extracción
        """
        processed_files = list(self.processed_dir.glob("*.txt"))

        return {
            "total_processed": len(processed_files),
            "processed_files": [str(f.name) for f in processed_files],
            "total_size_kb": sum(f.stat().st_size for f in processed_files) / 1024
        }