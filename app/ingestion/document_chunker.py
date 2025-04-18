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
