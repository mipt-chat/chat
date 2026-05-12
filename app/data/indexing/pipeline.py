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
from app.models import KnowledgeDocument

from .chunking import Chunk, split_text_into_chunks

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
        raise FileNotFoundError(f"Файл базы знаний не найден: {file_path}")

    content = file_path.read_text(encoding="utf-8").strip()
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
    from yaml import safe_load

    data = safe_load(file_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML должен содержать объект верхнего уровня: {file_path}")
    return data


def _resolve_import_path(current_yaml: Path, import_ref: str) -> Path:
    normalized = _normalize_path(import_ref)

    # Для ссылок вида "book/book.yaml" при текущем файле root/root.yaml
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
        raise FileNotFoundError(f"YAML конфиг базы знаний не найден: {resolved_yaml}")

    config = _load_yaml_config(resolved_yaml)
    documents: list[KnowledgeDocument] = []

    imports = config.get("imports", [])
    if imports is None:
        imports = []
    if not isinstance(imports, list):
        raise ValueError(f"Поле 'imports' должно быть списком: {resolved_yaml}")

    for item in imports:
        if not isinstance(item, str):
            raise ValueError(f"Элемент imports должен быть строкой: {resolved_yaml}")
        imported_yaml = _resolve_import_path(resolved_yaml, item)
        documents.extend(_collect_documents_from_yaml(imported_yaml, visited))

    docs = config.get("docs", {})
    if docs is None:
        docs = {}
    if not isinstance(docs, dict):
        raise ValueError(f"Поле 'docs' должно быть объектом: {resolved_yaml}")

    for doc_id, raw_meta in docs.items():
        if not isinstance(doc_id, str) or not isinstance(raw_meta, dict):
            raise ValueError(f"Некорректная запись docs в: {resolved_yaml}")
        source = raw_meta.get("source")
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"Для '{doc_id}' не задан корректный source в {resolved_yaml}")

        location_raw = raw_meta.get("location")
        location = location_raw if isinstance(location_raw, str) else None
        doc_path = _build_doc_path(resolved_yaml, location, source)
        if not doc_path.exists():
            raise FileNotFoundError(f"Документ '{doc_id}' не найден: {doc_path}")

        content = doc_path.read_text(encoding="utf-8").strip()
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
    """
    if not entry_path.exists():
        raise FileNotFoundError(f"Точка входа базы знаний не найдена: {entry_path}")

    suffix = entry_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        documents = _collect_documents_from_yaml(entry_path, visited=set())
    else:
        documents = [load_knowledge_document(entry_path)]

    if not documents:
        raise ValueError("Не найдено документов для индексации")

    return documents


def embed_chunks(model: SentenceTransformer, chunks: list[Chunk]) -> list[list[float]]:
    """
    Строит эмбеддинги чанков с e5-подготовкой текста.
    """
    prepared_texts = [add_e5_passage_prefix(chunk.text) for chunk in chunks]
    embeddings = model.encode(
        prepared_texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return embeddings.tolist()


def compute_document_hash(content: str) -> str:
    """
    Вычисляет стабильный hash содержимого документа.
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


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
    logger.info("Data indexing pipeline started")
    logger.info("Embedding model: %s", settings.embedding_model_name)

    knowledge_docs = load_knowledge_documents(settings.knowledge_base_file)
    logger.info("Knowledge documents loaded: %s", len(knowledge_docs))

    import chromadb

    client = chromadb.PersistentClient(path=settings.chroma_persist_directory)
    collection = client.get_or_create_collection(
        name=settings.chroma_collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    existing_ids_by_source, existing_hash_by_source = get_existing_documents_state(collection)

    current_sources = {doc.source_path for doc in knowledge_docs}
    stale_sources = set(existing_ids_by_source) - current_sources
    stale_ids = [
        chunk_id
        for source_path in stale_sources
        for chunk_id in existing_ids_by_source.get(source_path, [])
    ]
    if stale_ids:
        collection.delete(ids=stale_ids)
        logger.info("Removed %s stale chunks from deleted documents", len(stale_ids))

    docs_to_upsert = select_documents_for_incremental_upsert(
        knowledge_docs=knowledge_docs,
        existing_hash_by_source=existing_hash_by_source,
    )
    if not docs_to_upsert:
        logger.info("No document changes detected. Incremental indexing skipped.")
        return

    changed_ids = [
        chunk_id
        for doc in docs_to_upsert
        for chunk_id in existing_ids_by_source.get(doc.source_path, [])
    ]
    if changed_ids:
        collection.delete(ids=changed_ids)
        logger.info("Removed %s chunks for changed documents", len(changed_ids))

    chunk_rows = []
    for doc in docs_to_upsert:
        doc_chunks = split_text_into_chunks(
            text=doc.content,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        for chunk in doc_chunks:
            chunk_rows.append((doc, chunk))
    if not chunk_rows:
        logger.info("Changed documents have no chunks after splitting. Nothing to upsert.")
        return

    logger.info("Chunks prepared for upsert: %s", len(chunk_rows))

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(settings.embedding_model_name)
    logger.info("SentenceTransformer model initialized")

    embeddings = embed_chunks(model=model, chunks=[row[1] for row in chunk_rows])
    logger.info("Embeddings generated: %s", len(embeddings))

    ids = [f"{doc.source_path}:{chunk.index}" for doc, chunk in chunk_rows]
    documents = [chunk.text for _, chunk in chunk_rows]
    metadatas = [
        {
            "source_path": doc.source_path,
            "chunk_index": chunk.index,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
            "doc_id": doc.metadata.get("doc_id"),
            "doc_type": doc.metadata.get("doc_type"),
            "location": doc.metadata.get("location"),
            "yaml_path": doc.metadata.get("yaml_path"),
            "doc_hash": compute_document_hash(doc.content),
        }
        for doc, chunk in chunk_rows
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )
    logger.info(
        "Incremental indexing finished. Collection '%s': upserted %s docs, %s chunks",
        settings.chroma_collection_name,
        len(docs_to_upsert),
        len(chunk_rows),
    )


def main() -> None:
    """
    CLI entry point для запуска пайплайна.
    """
    run_indexing_pipeline()


if __name__ == "__main__":
    main()

