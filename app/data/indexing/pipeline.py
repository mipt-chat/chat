"""
Pipeline индексации базы знаний в ChromaDB.
Запуск: python -m app.data.indexing.pipeline
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.core import get_logger
from app.core.config import settings
from app.models import IndexedChunk, KnowledgeDocument

from .chroma_store import add_indexed_chunks
from .chunking import split_text_into_chunks

if TYPE_CHECKING:
    import chromadb
    from sentence_transformers import SentenceTransformer

logger = get_logger(__name__)


def add_e5_passage_prefix(text: str) -> str:
    """
    Добавляет обязательный префикс для индексации моделей семейства E5.
    """
    return f"passage: {text}"


def load_knowledge_document(file_path: Path) -> KnowledgeDocument:
    """
    Загружает единый текстовый документ базы знаний.
    """
    if not file_path.exists():
        logger.error("Knowledge document not found: %s", file_path)
        raise FileNotFoundError(f"Файл базы знаний не найден: {file_path}")

    try:
        raw = file_path.read_text(encoding="utf-8")
    except OSError as e:
        logger.error("Failed to read knowledge file %s: %s", file_path, e)
        raise
    except UnicodeDecodeError as e:
        logger.error("Invalid UTF-8 in knowledge file %s: %s", file_path, e)
        raise
    content = raw.strip()
    if not content:
        raise ValueError(f"Файл базы знаний пустой: {file_path}")

    return KnowledgeDocument(
        source_path=str(file_path),
        content=content,
        metadata={"source_path": str(file_path)},
    )


def _normalize_path(raw_path: str) -> str:
    return raw_path.replace("\\", "/")


def _load_yaml_config(file_path: Path) -> dict[str, Any]:
    from yaml import YAMLError, safe_load

    try:
        raw = file_path.read_text(encoding="utf-8")
    except OSError as e:
        logger.error("Failed to read YAML %s: %s", file_path, e)
        raise
    except UnicodeDecodeError as e:
        logger.error("Invalid UTF-8 in YAML %s: %s", file_path, e)
        raise
    try:
        data = safe_load(raw) or {}
    except YAMLError as e:
        logger.error("Invalid or corrupted YAML in %s: %s", file_path, e)
        raise
    if not isinstance(data, dict):
        logger.error("YAML root must be a mapping (object), got %s: %s", type(data).__name__, file_path)
        raise ValueError(f"YAML должен содержать объект верхнего уровня: {file_path}")
    return data


def _resolve_import_path(current_yaml: Path, import_ref: str) -> Path:
    normalized = _normalize_path(import_ref)

    # Для ссылок вида "book/book.yaml" при entrypoint в каталоге root/ (root.yaml)
    # поднимаемся на уровень выше, чтобы путь резолвился от папки базы.
    if current_yaml.parent.name == "root" and "/" in normalized:
        return (current_yaml.parent.parent / normalized).resolve()

    return (current_yaml.parent / normalized).resolve()


def _build_doc_path(yaml_file: Path, location: str | None, source: str) -> Path:
    """
    Резолвит путь к файлу документа.

    Поддерживает два варианта раскладки рядом с YAML:
    - вложенный: ``<yaml_dir>/<location>/<source>`` (как в манифесте);
    - плоский: ``<yaml_dir>/<source>`` (фактическая структура в репозитории).
    """
    source_path = Path(_normalize_path(source))
    base = yaml_file.parent

    if source_path.is_absolute():
        return source_path.resolve()

    if source_path.parent != Path("."):
        return (base / source_path).resolve()

    flat = (base / source_path.name).resolve()
    if location:
        nested = (base / _normalize_path(location) / source_path.name).resolve()
        if nested.exists():
            return nested
        if flat.exists():
            return flat
        return nested

    return flat


def _collect_documents_from_yaml(
    yaml_path: Path,
    visited: set[Path],
) -> list[KnowledgeDocument]:
    resolved_yaml = yaml_path.resolve()
    if resolved_yaml in visited:
        logger.debug("Skipping already visited yaml: %s", resolved_yaml)
        return []
    visited.add(resolved_yaml)

    if not resolved_yaml.exists():
        logger.error("YAML config not found: %s", resolved_yaml)
        raise FileNotFoundError(f"YAML конфиг базы знаний не найден: {resolved_yaml}")

    config = _load_yaml_config(resolved_yaml)
    documents: list[KnowledgeDocument] = []

    imports = config.get("imports", [])
    if imports is None:
        imports = []
    if not isinstance(imports, list):
        logger.error(
            "Invalid 'imports' in %s: expected a list, got %s",
            resolved_yaml,
            type(imports).__name__,
        )
        raise ValueError(f"Поле 'imports' должно быть списком: {resolved_yaml}")

    for item in imports:
        if not isinstance(item, str):
            logger.error(
                "Invalid imports entry in %s: expected string, got %s",
                resolved_yaml,
                type(item).__name__,
            )
            raise ValueError(f"Элемент imports должен быть строкой: {resolved_yaml}")
        imported_yaml = _resolve_import_path(resolved_yaml, item)
        documents.extend(_collect_documents_from_yaml(imported_yaml, visited))

    docs = config.get("docs", {})
    if docs is None:
        docs = {}
    if not isinstance(docs, dict):
        logger.error(
            "Invalid 'docs' in %s: expected a mapping, got %s",
            resolved_yaml,
            type(docs).__name__,
        )
        raise ValueError(f"Поле 'docs' должно быть объектом: {resolved_yaml}")

    logger.debug(
        "KB manifest %s: %s import(s), %s doc entries",
        resolved_yaml,
        len(imports),
        len(docs),
    )

    for doc_id, raw_meta in docs.items():
        if not isinstance(doc_id, str) or not isinstance(raw_meta, dict):
            logger.error("Invalid docs entry in %s: doc_id/meta types invalid", resolved_yaml)
            raise ValueError(f"Некорректная запись docs в: {resolved_yaml}")
        source = raw_meta.get("source")
        if not isinstance(source, str) or not source.strip():
            logger.error("Invalid or missing 'source' for doc %r in %s", doc_id, resolved_yaml)
            raise ValueError(f"Для '{doc_id}' не задан корректный source в {resolved_yaml}")

        location_raw = raw_meta.get("location")
        location = location_raw if isinstance(location_raw, str) else None
        doc_path = _build_doc_path(resolved_yaml, location, source)
        if not doc_path.exists():
            logger.error("Document '%s' not found at path: %s", doc_id, doc_path)
            raise FileNotFoundError(f"Документ '{doc_id}' не найден: {doc_path}")

        try:
            raw_content = doc_path.read_text(encoding="utf-8")
        except OSError as e:
            logger.error("Failed to read document '%s' (%s): %s", doc_id, doc_path, e)
            raise
        except UnicodeDecodeError as e:
            logger.error("Invalid UTF-8 in document '%s' (%s): %s", doc_id, doc_path, e)
            raise
        content = raw_content.strip()
        if not content:
            logger.warning("Skipping empty document '%s': %s", doc_id, doc_path)
            continue

        doc_metadata = {
            "source_path": str(doc_path),
            "doc_id": doc_id,
            "doc_type": raw_meta.get("type", "markdown"),
            "location": location,
            "yaml_path": str(resolved_yaml),
        }
        documents.append(
            KnowledgeDocument(
                source_path=str(doc_path),
                content=content,
                metadata=doc_metadata,
            )
        )

    return documents


def load_knowledge_documents(entry_path: Path) -> list[KnowledgeDocument]:
    """
    Загружает документы базы знаний из txt или yaml-entry.

    Для YAML: обход в глубину — ``imports`` (относительно каталога текущего yaml),
    затем словарь ``docs`` (поля ``source``, опционально ``location`` → открытие .txt).
    Повторный заход в один и тот же yaml по циклу импортов игнорируется (``visited``).
    """
    if not entry_path.exists():
        logger.error("Knowledge base entry path not found: %s", entry_path)
        raise FileNotFoundError(f"Точка входа базы знаний не найдена: {entry_path}")

    suffix = entry_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        documents = _collect_documents_from_yaml(entry_path, visited=set())
    else:
        documents = [load_knowledge_document(entry_path)]

    if not documents:
        logger.error("No indexable documents under knowledge base entry: %s", entry_path)
        raise ValueError("Не найдено документов для индексации")

    return documents


def embed_chunks(model: SentenceTransformer, chunks: list[IndexedChunk]) -> list[list[float]]:
    """
    Строит эмбеддинги чанков с e5-подготовкой текста.
    """
    prepared_texts = [add_e5_passage_prefix(chunk.text) for chunk in chunks]
    try:
        embeddings = model.encode(
            prepared_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
    except Exception as e:
        logger.error(
            "model.encode failed (%s chunks); check GPU/memory and input length: %s",
            len(chunks),
            e,
        )
        raise
    return embeddings.tolist()


def compute_document_hash(content: str) -> str:
    """
    Вычисляет стабильный hash содержимого документа.
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _chroma_metadata_for_index(
    doc: KnowledgeDocument,
    chunk_index: int,
) -> dict[str, str | int | float | bool]:
    """
    Собирает metadata для Chroma без значений None (Chroma их не принимает).
    """
    meta: dict[str, str | int | float | bool] = {
        "source_path": doc.source_path,
        "chunk_index": chunk_index,
        "doc_hash": compute_document_hash(doc.content),
    }
    for key in ("doc_id", "doc_type", "location", "yaml_path"):
        val = doc.metadata.get(key)
        if val is not None:
            meta[key] = val if isinstance(val, str | int | float | bool) else str(val)
    return meta


def build_indexed_chunks_for_document(
    doc: KnowledgeDocument,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[IndexedChunk]:
    """
    Нарезает документ и собирает общие модели IndexedChunk для Chroma / эмбеддера.
    """
    raw_chunks = split_text_into_chunks(
        text=doc.content,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return [
        IndexedChunk(
            chunk_id=f"{doc.source_path}:{c.index}",
            text=c.text,
            chunk_index=c.index,
            metadata=_chroma_metadata_for_index(doc, c.index),
        )
        for c in raw_chunks
    ]


def get_existing_documents_state(
    collection: chromadb.Collection,
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """
    Возвращает состояние уже проиндексированных документов:
    - source_path -> list(chunk ids)
    - source_path -> doc_hash
    """
    existing = collection.get(include=["metadatas"])
    ids = existing.get("ids", []) or []
    metadatas = existing.get("metadatas", []) or []

    ids_by_source: dict[str, list[str]] = {}
    hash_by_source: dict[str, str] = {}

    for chunk_id, metadata in zip(ids, metadatas, strict=False):
        if not isinstance(metadata, dict):
            continue
        source_path = metadata.get("source_path")
        if not isinstance(source_path, str) or not source_path:
            continue
        ids_by_source.setdefault(source_path, []).append(chunk_id)

        doc_hash = metadata.get("doc_hash")
        if isinstance(doc_hash, str) and doc_hash and source_path not in hash_by_source:
            hash_by_source[source_path] = doc_hash

    return ids_by_source, hash_by_source


def select_documents_for_incremental_upsert(
    knowledge_docs: list[KnowledgeDocument],
    existing_hash_by_source: dict[str, str],
) -> list[KnowledgeDocument]:
    """
    Оставляет только новые/изменившиеся документы.
    """
    docs_to_upsert: list[KnowledgeDocument] = []
    for doc in knowledge_docs:
        current_hash = compute_document_hash(doc.content)
        existing_hash = existing_hash_by_source.get(doc.source_path)
        if current_hash != existing_hash:
            docs_to_upsert.append(doc)
    return docs_to_upsert


def run_indexing_pipeline() -> None:
    """
    Выполняет полный цикл: загрузка -> чанкинг -> эмбеддинг -> запись в ChromaDB.
    """
    logger.info("Knowledge base indexing started")
    logger.info("Embedding model: %s", settings.embedding_model_name)

    knowledge_docs = load_knowledge_documents(settings.knowledge_base_file)
    total_chunks_if_full_rebuild = sum(
        len(
            build_indexed_chunks_for_document(
                doc,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
        )
        for doc in knowledge_docs
    )
    logger.info(
        "Knowledge documents loaded: %s; total chunks if full reindex: %s",
        len(knowledge_docs),
        total_chunks_if_full_rebuild,
    )

    import chromadb

    try:
        client = chromadb.PersistentClient(path=settings.chroma_persist_directory)
        collection = client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
    except Exception as e:
        logger.error(
            "Failed to open ChromaDB at %r collection %r: %s",
            settings.chroma_persist_directory,
            settings.chroma_collection_name,
            e,
        )
        raise

    try:
        existing_ids_by_source, existing_hash_by_source = get_existing_documents_state(collection)
    except Exception as e:
        logger.error("Failed to read existing state from ChromaDB: %s", e)
        raise

    existing_chunk_count = collection.count()
    logger.info(
        "ChromaDB ready: persist=%s collection=%s existing_chunks=%s",
        settings.chroma_persist_directory,
        settings.chroma_collection_name,
        existing_chunk_count,
    )

    current_sources = {doc.source_path for doc in knowledge_docs}
    stale_sources = set(existing_ids_by_source) - current_sources
    stale_ids = [
        chunk_id
        for source_path in stale_sources
        for chunk_id in existing_ids_by_source.get(source_path, [])
    ]
    if stale_ids:
        try:
            collection.delete(ids=stale_ids)
        except Exception as e:
            logger.error("ChromaDB delete (stale chunks) failed: %s", e)
            raise
        logger.info("Removed %s stale chunks from deleted documents", len(stale_ids))

    docs_to_upsert = select_documents_for_incremental_upsert(
        knowledge_docs=knowledge_docs,
        existing_hash_by_source=existing_hash_by_source,
    )
    if not docs_to_upsert:
        logger.info(
            "Incremental indexing skipped: no document text changes "
            "(%s documents in KB, %s chunks currently in ChromaDB)",
            len(knowledge_docs),
            collection.count(),
        )
        return

    changed_ids = [
        chunk_id
        for doc in docs_to_upsert
        for chunk_id in existing_ids_by_source.get(doc.source_path, [])
    ]
    if changed_ids:
        try:
            collection.delete(ids=changed_ids)
        except Exception as e:
            logger.error("ChromaDB delete (changed documents) failed: %s", e)
            raise
        logger.info("Removed %s chunks for changed documents", len(changed_ids))

    indexed_chunks: list[IndexedChunk] = []
    for doc in docs_to_upsert:
        indexed_chunks.extend(
            build_indexed_chunks_for_document(
                doc,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
        )
    if not indexed_chunks:
        logger.info("Changed documents have no chunks after splitting. Nothing to upsert.")
        return

    logger.info(
        "Prepared %s chunks from %s documents for upsert to ChromaDB",
        len(indexed_chunks),
        len(docs_to_upsert),
    )

    from sentence_transformers import SentenceTransformer

    try:
        model = SentenceTransformer(settings.embedding_model_name)
    except Exception as e:
        logger.error(
            "Failed to load embedding model %r (network, disk, or HF Hub): %s",
            settings.embedding_model_name,
            e,
        )
        raise

    logger.info("SentenceTransformer model initialized")
    logger.info(
        "Computing embeddings for %s chunks (no progress bar; on CPU this may take several minutes)",
        len(indexed_chunks),
    )

    embeddings = embed_chunks(model=model, chunks=indexed_chunks)
    logger.info("Embeddings generated: %s", len(embeddings))

    try:
        add_indexed_chunks(collection, indexed_chunks, embeddings)
    except Exception as e:
        logger.error("ChromaDB collection.add failed (%s vectors): %s", len(indexed_chunks), e)
        raise

    logger.info(
        "Successfully saved to ChromaDB: collection=%r added_chunks=%s documents_upserted=%s",
        settings.chroma_collection_name,
        len(indexed_chunks),
        len(docs_to_upsert),
    )


def main() -> None:
    """
    CLI entry point для запуска пайплайна.
    """
    run_indexing_pipeline()


if __name__ == "__main__":
    main()

