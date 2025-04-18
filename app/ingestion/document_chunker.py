# app/ingestion/document_chunker.py

import logging
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Importamos LlamaIndex para usar sus utilidades de procesamiento de documentos
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
import tiktoken

logger = logging.getLogger(__name__)


class DocumentChunker:
    """
    Divide documentos de texto en fragmentos óptimos para procesamiento RAG.

    Esta clase implementa funcionalidades para dividir textos en fragmentos
    de tamaño configurable, con opciones de solapamiento y preservación
    de integridad semántica.
    """

    def __init__(
            self,
            chunk_size: int = 1024,
            chunk_overlap: int = 200,
            tokenizer_name: str = "cl100k_base"  # Tokenizador por defecto de GPT-4
    ):
        """
        Inicializa el fragmentador de documentos.

        Args:
            chunk_size: Tamaño máximo de cada fragmento en tokens
            chunk_overlap: Cantidad de tokens que se solapan entre fragmentos
            tokenizer_name: Nombre del tokenizador a utilizar
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer_name = tokenizer_name

        # Inicializar el tokenizador de tiktoken
        try:
            self.tokenizer = tiktoken.get_encoding(tokenizer_name)
            logger.info(f"Tokenizador {tokenizer_name} inicializado correctamente")
        except Exception as e:
            logger.error(f"Error al inicializar el tokenizador {tokenizer_name}: {str(e)}")
            raise

        # Inicializar el divisor de LlamaIndex
        self.splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            tokenizer=self.tokenizer.encode
        )

        logger.info(f"DocumentChunker inicializado con tamaño de fragmento={chunk_size}, "
                    f"solapamiento={chunk_overlap}")

    def count_tokens(self, text: str) -> int:
        """
        Cuenta el número de tokens en un texto dado.

        Args:
            text: Texto a analizar

        Returns:
            Número de tokens en el texto
        """
        if not text:
            return 0

        try:
            tokens = self.tokenizer.encode(text)
            return len(tokens)
        except Exception as e:
            logger.error(f"Error al contar tokens: {str(e)}")
            raise

    def load_document(self, file_path: str) -> str:
        """
        Carga un documento de texto desde un archivo.

        Args:
            file_path: Ruta al archivo de texto

        Returns:
            Contenido del archivo como una cadena de texto
        """
        file_path = Path(file_path)

        if not file_path.exists():
            error_msg = f"No se encontró el archivo: {file_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            logger.info(f"Documento cargado: {file_path.name} ({self.count_tokens(content)} tokens)")
            return content
        except Exception as e:
            logger.error(f"Error al cargar el documento {file_path}: {str(e)}")
            raise

    def split_text(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Divide un texto en fragmentos óptimos para RAG.

        Args:
            text: Texto a dividir en fragmentos
            metadata: Metadatos opcionales para asociar con los fragmentos

        Returns:
            Lista de diccionarios, cada uno conteniendo un fragmento y sus metadatos
        """
        if not text:
            logger.warning("Se intentó dividir un texto vacío")
            return []

        try:
            # Crear un documento LlamaIndex
            llama_doc = Document(text=text, metadata=metadata or {})

            # Usar el splitter para dividir el documento
            nodes = self.splitter.get_nodes_from_documents([llama_doc])

            # Convertir nodos a diccionarios
            chunks = []
            for i, node in enumerate(nodes):
                chunk = {
                    "chunk_id": i,
                    "text": node.text,
                    "token_count": self.count_tokens(node.text),
                    "metadata": {
                        **node.metadata,
                        "chunk_index": i,
                        "total_chunks": len(nodes)
                    }
                }
                chunks.append(chunk)

            logger.info(f"Texto dividido en {len(chunks)} fragmentos")

            # Verificar integridad de los fragmentos
            total_tokens = sum(chunk["token_count"] for chunk in chunks)
            logger.info(f"Total de tokens en fragmentos: {total_tokens}")

            return chunks

        except Exception as e:
            logger.error(f"Error al dividir el texto en fragmentos: {str(e)}")
            raise
