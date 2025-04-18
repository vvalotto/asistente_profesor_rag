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

        # Inicializar metadatos si no se proporcionaron
        metadata = metadata or {}

        try:
            # Crear un documento LlamaIndex
            llama_doc = Document(text=text, metadata=metadata)

            # Intentar usar el splitter para dividir el documento
            try:
                nodes = self.splitter.get_nodes_from_documents([llama_doc])
            except ValueError as e:
                # Si el error es que los metadatos son demasiado grandes
                if "Metadata length" in str(e) and "is longer than chunk size" in str(e):
                    logger.warning(f"Los metadatos son demasiado grandes para el tamaño de fragmento: {str(e)}")
                    logger.warning("Usando método alternativo para fragmentar el texto")

                    # Calcular cuántos tokens del tamaño del fragmento están disponibles para texto
                    metadata_tokens = self.count_tokens(str(metadata))
                    available_tokens = max(self.chunk_size - metadata_tokens,
                                           10)  # Asegurar al menos 10 tokens para texto

                    # Fragmentar el texto manualmente
                    return self._manual_split_text(text, metadata, available_tokens)
                else:
                    # Si es un error diferente, propagar
                    raise

            # Si no se generaron nodos (texto muy corto), crear uno manualmente
            if not nodes:
                logger.warning("El splitter no generó nodos, creando un fragmento manualmente")
                # Crear un nodo manual con el texto completo
                from llama_index.core.schema import TextNode
                node = TextNode(text=text, metadata=metadata)
                nodes = [node]

            # Convertir nodos a diccionarios
            chunks = []
            for i, node in enumerate(nodes):
                chunk = {
                    "chunk_id": i,
                    "text": node.text,
                    "token_count": self.count_tokens(node.text),
                    "metadata": {
                        **(node.metadata if hasattr(node, 'metadata') else {}),
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

    def _manual_split_text(self, text: str, metadata: Dict, available_tokens: int) -> List[Dict]:
        """
        Método alternativo para dividir texto cuando el splitter de LlamaIndex falla.

        Args:
            text: Texto a dividir
            metadata: Metadatos para cada fragmento
            available_tokens: Tokens disponibles para texto en cada fragmento

        Returns:
            Lista de diccionarios con fragmentos
        """
        # Dividir el texto en oraciones primero
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = ""
        current_tokens = 0
        chunk_id = 0

        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)

            # Si la oración es muy grande, podríamos subdividirla aún más
            if sentence_tokens > available_tokens:
                # Si ya tenemos contenido en el fragmento actual, guardarlo primero
                if current_chunk:
                    chunks.append({
                        "chunk_id": chunk_id,
                        "text": current_chunk,
                        "token_count": current_tokens,
                        "metadata": {
                            **metadata,
                            "chunk_index": chunk_id,
                            "manual_split": True
                        }
                    })
                    chunk_id += 1
                    current_chunk = ""
                    current_tokens = 0

                # Dividir la oración en fragmentos más pequeños
                words = sentence.split()
                sentence_part = ""
                part_tokens = 0

                for word in words:
                    word_tokens = self.count_tokens(word + " ")

                    if part_tokens + word_tokens <= available_tokens:
                        sentence_part += word + " "
                        part_tokens += word_tokens
                    else:
                        # Guardar fragmento parcial
                        chunks.append({
                            "chunk_id": chunk_id,
                            "text": sentence_part.strip(),
                            "token_count": part_tokens,
                            "metadata": {
                                **metadata,
                                "chunk_index": chunk_id,
                                "manual_split": True
                            }
                        })
                        chunk_id += 1
                        sentence_part = word + " "
                        part_tokens = word_tokens

                # Guardar último fragmento parcial de la oración
                if sentence_part:
                    chunks.append({
                        "chunk_id": chunk_id,
                        "text": sentence_part.strip(),
                        "token_count": part_tokens,
                        "metadata": {
                            **metadata,
                            "chunk_index": chunk_id,
                            "manual_split": True
                        }
                    })
                    chunk_id += 1

            # La oración cabe en el fragmento actual o en uno nuevo
            elif current_tokens + sentence_tokens <= available_tokens:
                # Añadir al fragmento actual
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                current_tokens += sentence_tokens
            else:
                # La oración no cabe, guardar el fragmento actual y empezar uno nuevo
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": current_chunk,
                    "token_count": current_tokens,
                    "metadata": {
                        **metadata,
                        "chunk_index": chunk_id,
                        "manual_split": True
                    }
                })
                chunk_id += 1
                current_chunk = sentence
                current_tokens = sentence_tokens

        # Guardar el último fragmento si hay contenido
        if current_chunk:
            chunks.append({
                "chunk_id": chunk_id,
                "text": current_chunk,
                "token_count": current_tokens,
                "metadata": {
                    **metadata,
                    "chunk_index": chunk_id,
                    "manual_split": True,
                    "total_chunks": chunk_id + 1
                }
            })

        # Actualizar el total de fragmentos en todos los metadatos
        total_chunks = len(chunks)
        for chunk in chunks:
            chunk["metadata"]["total_chunks"] = total_chunks

        logger.info(f"Texto dividido manualmente en {len(chunks)} fragmentos")
        return chunks

    def process_file(self, file_path: str, output_dir: Optional[str] = None,
                     metadata: Optional[Dict] = None) -> Tuple[List[Dict], Dict]:
        """
        Procesa un archivo completo: carga, divide en fragmentos y opcionalmente guarda.

        Args:
            file_path: Ruta al archivo a procesar
            output_dir: Directorio donde guardar los fragmentos (opcional)
            metadata: Metadatos adicionales para asociar con los fragmentos

        Returns:
            Tupla con la lista de fragmentos y un diccionario de estadísticas
        """
        file_path = Path(file_path)

        # Generar metadatos base del archivo
        import os
        import time

        # Obtener la fecha de creación del archivo usando os.path.getctime
        creation_time = time.ctime(os.path.getctime(file_path))

        base_metadata = {
            "source": str(file_path),
            "filename": file_path.name,
            "processed_date": str(creation_time),
        }

        # Combinar con metadatos proporcionados
        if metadata:
            base_metadata.update(metadata)

        # Cargar el documento
        content = self.load_document(file_path)

        # Dividir en fragmentos
        chunks = self.split_text(content, base_metadata)

        # Generar estadísticas
        stats = {
            "source_file": str(file_path),
            "total_chunks": len(chunks),
            "total_tokens": sum(chunk["token_count"] for chunk in chunks),
            "avg_chunk_size": sum(chunk["token_count"] for chunk in chunks) / len(chunks) if chunks else 0,
            "min_chunk_size": min(chunk["token_count"] for chunk in chunks) if chunks else 0,
            "max_chunk_size": max(chunk["token_count"] for chunk in chunks) if chunks else 0,
        }

        # Guardar fragmentos si se especificó un directorio de salida
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)

            # Crear un subdirectorio específico para este archivo
            file_dir = output_dir / file_path.stem
            file_dir.mkdir(exist_ok=True)

            # Guardar cada fragmento en un archivo separado
            for chunk in chunks:
                chunk_file = file_dir / f"chunk_{chunk['chunk_id']:04d}.json"
                with open(chunk_file, 'w', encoding='utf-8') as f:
                    import json
                    json.dump(chunk, f, ensure_ascii=False, indent=2)

            # Guardar estadísticas
            stats_file = file_dir / "stats.json"
            with open(stats_file, 'w', encoding='utf-8') as f:
                import json
                json.dump(stats, f, ensure_ascii=False, indent=2)

            logger.info(f"Fragmentos guardados en: {file_dir}")

        return chunks, stats

    def process_directory(self, input_dir: str, output_dir: str,
                          file_pattern: str = "*.txt",
                          metadata: Optional[Dict] = None) -> Dict:
        """
        Procesa todos los archivos que coinciden con un patrón en un directorio.

        Args:
            input_dir: Directorio que contiene los archivos a procesar
            output_dir: Directorio donde guardar los fragmentos
            file_pattern: Patrón glob para filtrar archivos (por defecto: "*.txt")
            metadata: Metadatos adicionales comunes para asociar con todos los fragmentos

        Returns:
            Diccionario con estadísticas agregadas
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        if not input_dir.exists() or not input_dir.is_dir():
            error_msg = f"El directorio de entrada no existe: {input_dir}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # Crear directorio de salida si no existe
        output_dir.mkdir(exist_ok=True, parents=True)

        # Encontrar todos los archivos que coincidan con el patrón
        files = list(input_dir.glob(file_pattern))

        if not files:
            logger.warning(f"No se encontraron archivos con el patrón {file_pattern} en {input_dir}")
            return {"processed_files": 0, "total_chunks": 0, "total_tokens": 0}

        logger.info(f"Procesando {len(files)} archivos de {input_dir}")

        # Procesar cada archivo
        all_stats = []
        errors = []

        for file in files:
            try:
                # Si hay metadatos específicos para este archivo, se podrían determinar aquí
                file_metadata = {} if metadata is None else metadata.copy()

                # Procesar el archivo
                chunks, stats = self.process_file(file, output_dir, file_metadata)

                # Si no hay chunks pero no hubo excepción, aún así consideramos el archivo como procesado
                all_stats.append(stats)

                logger.info(f"Archivo procesado: {file.name} - {stats['total_chunks']} fragmentos")

            except Exception as e:
                error_msg = f"Error al procesar el archivo {file}: {str(e)}"
                logger.error(error_msg)
                errors.append({"file": str(file), "error": str(e)})
                continue

        # Generar estadísticas agregadas
        aggregate_stats = {
            "processed_files": len(all_stats),
            "failed_files": len(files) - len(all_stats),
            "total_chunks": sum(s["total_chunks"] for s in all_stats),
            "total_tokens": sum(s["total_tokens"] for s in all_stats),
            "avg_chunks_per_file": sum(s["total_chunks"] for s in all_stats) / len(all_stats) if all_stats else 0,
            "avg_tokens_per_chunk": (sum(s["total_tokens"] for s in all_stats) /
                                     sum(s["total_chunks"] for s in all_stats)) if sum(
                s["total_chunks"] for s in all_stats) > 0 else 0,
            "files": [s["source_file"] for s in all_stats],
            "errors": errors  # Añadir información de errores
        }

        # Guardar estadísticas agregadas
        stats_file = output_dir / "aggregate_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            import json
            json.dump(aggregate_stats, f, ensure_ascii=False, indent=2)

        logger.info(f"Procesamiento completo. Estadísticas guardadas en {stats_file}")
        logger.info(f"Archivos procesados: {aggregate_stats['processed_files']}, "
                    f"Archivos fallidos: {aggregate_stats['failed_files']}")

        return aggregate_stats