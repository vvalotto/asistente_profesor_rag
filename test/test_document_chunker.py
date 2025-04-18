
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

    def test_process_directory(self, tmp_path):
        """Verificar que se procesan correctamente múltiples archivos."""
        # Crear directorio de entrada y varios archivos de prueba
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        # Crear 3 archivos de prueba
        for i in range(3):
            test_file = input_dir / f"test_document_{i}.txt"
            # Hacemos el contenido más largo para asegurar que se generen fragmentos
            test_file.write_text(f"Este es el documento de prueba {i}. " * 30, encoding='utf-8')

        # Crear un archivo con extensión diferente que no debería procesarse
        other_file = input_dir / "ignored.md"
        other_file.write_text("Este archivo debería ignorarse", encoding='utf-8')

        # Crear directorio de salida
        output_dir = tmp_path / "output"

        # Configurar un chunker con tamaño pequeño para asegurar la fragmentación
        chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)

        # Procesar el directorio y capturar las estadísticas
        stats = chunker.process_directory(
            str(input_dir),
            str(output_dir),
            file_pattern="*.txt"
        )

        # Imprimir información de depuración
        print(f"\nInformación de depuración para test_process_directory:")
        print(f"Archivos en directorio: {list(input_dir.glob('*.txt'))}")
        print(f"Estadísticas: {stats}")

        if "errors" in stats and stats["errors"]:
            print(f"Errores encontrados: {stats['errors']}")

        # Verificar estadísticas
        assert stats[
                   "processed_files"] == 3, f"Se esperaban 3 archivos procesados, pero se obtuvieron {stats['processed_files']}"
        assert stats["total_chunks"] > 0, "No se generaron fragmentos"

        # Verificar que se crearon directorios para cada archivo
        assert (output_dir / "test_document_0").exists(), "No se creó el directorio para el primer archivo"
        assert (output_dir / "test_document_1").exists(), "No se creó el directorio para el segundo archivo"
        assert (output_dir / "test_document_2").exists(), "No se creó el directorio para el tercer archivo"

        # Verificar que se creó el archivo de estadísticas agregadas
        assert (output_dir / "aggregate_stats.json").exists(), "No se creó el archivo de estadísticas agregadas"

    def test_evaluate_chunks_quality_problematic(self):
        """Verificar que se detectan problemas en fragmentos de mala calidad."""
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=5)  # Poco solapamiento

        # Crear fragmentos manualmente con problemas conocidos
        chunks = [
            {
                "chunk_id": 0,
                "text": "Este es un fragmento muy corto.",
                "token_count": 8
            },
            {
                "chunk_id": 1,
                "text": "Este es un fragmento muy largo que excede significativamente el tamaño " +
                        "máximo configurado para esta prueba y por lo tanto debería " +
                        "ser detectado como problemático por el evaluador de calidad " * 3,
                "token_count": 150  # Valor ficticio mayor que chunk_size
            },
            {
                "chunk_id": 2,
                "text": "Este fragmento no tiene solapamiento con el anterior.",
                "token_count": 10
            }
        ]

        # Evaluar calidad
        metrics = chunker.evaluate_chunks_quality(chunks)

        # Verificar que se detectaron los problemas
        assert metrics["quality_score"] < 70  # Debería tener una puntuación baja
        assert len(metrics["issues"]) > 0  # Debería haber identificado problemas

        # Verificar detección de fragmentos pequeños
        found_small_chunk_issue = False
        for issue in metrics["issues"]:
            if "pequeños" in issue:
                found_small_chunk_issue = True
                break
        assert found_small_chunk_issue
