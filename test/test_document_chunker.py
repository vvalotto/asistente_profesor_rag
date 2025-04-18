
import os
import sys
import logging
import pytest
from pathlib import Path
from app.ingestion.document_chunker import DocumentChunker

# Asegurarse de que la aplicación esté en el path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

    def test_count_tokens(self):
        """Verificar que el conteo de tokens funciona correctamente."""
        chunker = DocumentChunker()

        # Prueba con texto vacío
        assert chunker.count_tokens("") == 0

        # Prueba con texto simple
        assert chunker.count_tokens("Hello, world!") > 0

        # Prueba con texto largo
        long_text = "This is a longer text that should be tokenized " * 20
        token_count = chunker.count_tokens(long_text)
        assert token_count > 100

        # Verificar que el tokenizador funciona correctamente
        # El número exacto de tokens depende del tokenizador, pero podemos hacer
        # pruebas aproximadas basadas en el comportamiento esperado
        text1 = "Hello world"
        text2 = "Hello world " * 2
        assert chunker.count_tokens(text2) > chunker.count_tokens(text1)

    def test_load_document(self, tmp_path):
        """Verificar que se puede cargar un documento correctamente."""
        # Crear un archivo temporal para las pruebas
        test_content = "Este es un documento de prueba para verificar la carga de archivos."
        test_file = tmp_path / "test_document.txt"
        test_file.write_text(test_content, encoding='utf-8')

        chunker = DocumentChunker()
        loaded_content = chunker.load_document(str(test_file))

        # Verificar que el contenido se cargó correctamente
        assert loaded_content == test_content

    def test_load_document_not_found(self):
        """Verificar que se lanza una excepción cuando el archivo no existe."""
        chunker = DocumentChunker()

        with pytest.raises(FileNotFoundError):
            chunker.load_document("archivo_inexistente.txt")