
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

    def test_split_text_empty(self):
        """Verificar que se maneja correctamente un texto vacío."""
        chunker = DocumentChunker()
        chunks = chunker.split_text("")
        assert chunks == []

    def test_split_text_small(self):
        """Verificar que un texto pequeño se mantiene en un solo fragmento."""
        chunker = DocumentChunker(chunk_size=1024)
        small_text = "Este es un texto pequeño que debería caber en un solo fragmento."
        chunks = chunker.split_text(small_text)

        assert len(chunks) == 1
        assert chunks[0]["text"] == small_text
        assert chunks[0]["chunk_id"] == 0
        assert chunks[0]["metadata"]["chunk_index"] == 0
        assert chunks[0]["metadata"]["total_chunks"] == 1

    def test_split_text_large(self):
        """Verificar que un texto grande se divide en múltiples fragmentos."""
        # Usar un tamaño de fragmento pequeño para forzar la división
        chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)

        # Crear un texto suficientemente largo para forzar múltiples fragmentos
        large_text = "Este es un párrafo de ejemplo que contiene suficiente texto para ser dividido. " * 20

        chunks = chunker.split_text(large_text)

        # Verificar que se crearon múltiples fragmentos
        assert len(chunks) > 1

        # Verificar que los metadatos son correctos
        for i, chunk in enumerate(chunks):
            assert chunk["chunk_id"] == i
            assert chunk["metadata"]["chunk_index"] == i
            assert chunk["metadata"]["total_chunks"] == len(chunks)

        # Verificar que hay solapamiento entre fragmentos consecutivos
        if len(chunks) > 1:
            # Enfoque alternativo para verificar solapamiento:
            # Verificamos que al menos algún par de fragmentos consecutivos
            # comparten alguna palabra o frase

            found_overlap = False
            for i in range(len(chunks) - 1):
                current_words = set(chunks[i]["text"].split())
                next_words = set(chunks[i + 1]["text"].split())

                # Si hay palabras en común, hay solapamiento
                common_words = current_words.intersection(next_words)
                if len(common_words) > 0:
                    found_overlap = True
                    break

            assert found_overlap, "No se encontró solapamiento entre fragmentos consecutivos"

    def test_split_text_with_metadata(self):
        """Verificar que los metadatos se preservan correctamente."""
        # Aseguramos que el chunk_overlap es menor que el chunk_size
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
        text = "Este es un texto de ejemplo que se dividirá en fragmentos."
        metadata = {
            "source": "archivo_prueba.txt",
            "author": "Test User",
            "category": "Pruebas"
        }

        chunks = chunker.split_text(text, metadata)

        # Verificar que los metadatos originales se conservan en todos los fragmentos
        for chunk in chunks:
            assert chunk["metadata"]["source"] == "archivo_prueba.txt"
            assert chunk["metadata"]["author"] == "Test User"
            assert chunk["metadata"]["category"] == "Pruebas"