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
        Metodo alternativo para dividir texto cuando el splitter de LlamaIndex falla.

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

    def evaluate_chunks_quality(self, chunks: List[Dict]) -> Dict:
        """
        Evalúa la calidad de los fragmentos generados.

        Esta función analiza varias métricas para determinar si los fragmentos
        son óptimos para su uso en un sistema RAG:
        - Distribución de tamaños
        - Preservación de párrafos/frases
        - Solapamiento efectivo

        Args:
            chunks: Lista de fragmentos generados por split_text o process_file

        Returns:
            Diccionario con métricas de calidad
        """
        if not chunks:
            logger.warning("No hay fragmentos para evaluar")
            return {"quality_score": 0, "issues": ["No hay fragmentos para evaluar"]}

        # Inicializar métricas
        metrics = {
            "total_chunks": len(chunks),
            "size_distribution": {},
            "semantic_integrity": {
                "sentence_breaks": 0,  # Oraciones cortadas entre fragmentos
                "paragraph_preservation": 0  # Fragmentos que contienen párrafos completos
            },
            "overlap_metrics": {
                "effective_overlaps": 0,  # Fragmentos con solapamiento efectivo
                "insufficient_overlaps": 0  # Fragmentos con solapamiento insuficiente
            },
            "issues": []
        }

        # Analizar distribución de tamaños
        token_counts = [chunk["token_count"] for chunk in chunks]
        avg_size = sum(token_counts) / len(token_counts)
        max_size = max(token_counts)
        min_size = min(token_counts)

        metrics["size_distribution"] = {
            "average": avg_size,
            "max": max_size,
            "min": min_size,
            "std_dev": (sum((x - avg_size) ** 2 for x in token_counts) / len(token_counts)) ** 0.5
        }

        # Verificar si hay fragmentos muy pequeños o muy grandes
        if min_size < self.chunk_size * 0.5:
            metrics["issues"].append(f"Hay fragmentos muy pequeños (mínimo: {min_size} tokens)")

        if max_size > self.chunk_size * 1.1:
            metrics["issues"].append(f"Hay fragmentos que exceden el tamaño máximo (máximo: {max_size} tokens)")

        # Análisis de integridad semántica y solapamiento
        for i in range(len(chunks) - 1):
            current_chunk = chunks[i]["text"]
            next_chunk = chunks[i + 1]["text"]

            # Verificar si hay oraciones cortadas (aproximado mediante puntuación)
            if not current_chunk.endswith((".", "!", "?", ":", ";", "»", '"', "'")):
                metrics["semantic_integrity"]["sentence_breaks"] += 1

            # Verificar solapamiento efectivo
            found_overlap = False

            # Buscar las últimas N palabras del fragmento actual en el siguiente
            last_words = " ".join(current_chunk.split()[-10:])  # Últimas 10 palabras
            if any(word in next_chunk for word in last_words.split()):
                metrics["overlap_metrics"]["effective_overlaps"] += 1
                found_overlap = True

            if not found_overlap:
                metrics["overlap_metrics"]["insufficient_overlaps"] += 1

        # Calcular puntuación general de calidad (0-100)
        # 50% basado en distribución de tamaños
        # 30% basado en integridad semántica
        # 20% basado en solapamiento efectivo

        size_score = 50 * (1 - metrics["size_distribution"]["std_dev"] / self.chunk_size)
        size_score = max(0, min(50, size_score))

        semantic_score = 30 * (1 - metrics["semantic_integrity"]["sentence_breaks"] / len(chunks))
        semantic_score = max(0, min(30, semantic_score))

        overlap_score = 0
        if len(chunks) > 1:
            overlap_score = 20 * (metrics["overlap_metrics"]["effective_overlaps"] / (len(chunks) - 1))
        overlap_score = max(0, min(20, overlap_score))

        metrics["quality_score"] = size_score + semantic_score + overlap_score

        # Agregar interpretación de la puntuación
        if metrics["quality_score"] >= 85:
            metrics["quality_interpretation"] = "Excelente calidad de fragmentación"
        elif metrics["quality_score"] >= 70:
            metrics["quality_interpretation"] = "Buena calidad, puede usarse"
        elif metrics["quality_score"] >= 50:
            metrics["quality_interpretation"] = "Calidad aceptable, pero considere ajustar parámetros"
        else:
            metrics["quality_interpretation"] = "Calidad insuficiente, se recomienda ajustar parámetros"

        logger.info(f"Evaluación de calidad: {metrics['quality_score']}/100 - {metrics['quality_interpretation']}")

        return metrics