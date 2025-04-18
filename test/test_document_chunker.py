# tests/test_document_chunker.py

import pytest
import os
import sys
import logging
from pathlib import Path

# Asegurarse de que la aplicación esté en el path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ingestion.document_chunker import DocumentChunker

# Configurar logging para pruebas
logging.basicConfig(level=logging.INFO)


class TestDocumentChunker:
    """Pruebas unitarias para la clase DocumentChunker."""

    def test_initialization(self):
        """Verificar que la clase se inicializa correctamente con valores predeterminados."""
        chunker = DocumentChunker()
        assert chunker.chunk_size == 1024
        assert chunker.chunk_overlap == 200
        assert chunker.tokenizer_name == "cl100k_base"
        assert chunker.tokenizer is not None
        assert chunker.splitter is not None

    def test_custom_initialization(self):
        """Verificar que la clase se inicializa correctamente con valores personalizados."""
        chunker = DocumentChunker(chunk_size=512, chunk_overlap=100, tokenizer_name="cl100k_base")
        assert chunker.chunk_size == 512
        assert chunker.chunk_overlap == 100
        assert chunker.tokenizer_name == "cl100k_base"
        assert chunker.tokenizer is not None
        assert chunker.splitter is not None

    def test_invalid_tokenizer(self):
        """Verificar que se lanza una excepción con un tokenizador inválido."""
        with pytest.raises(Exception):
            chunker = DocumentChunker(tokenizer_name="invalid_tokenizer_name")