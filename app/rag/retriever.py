from __future__ import annotations

from functools import lru_cache
from typing import Any

from app.core import get_logger
from app.core.config import settings
from app.core.exceptions import EmbeddingError, RetrievalError
from app.models.knowledge import RetrievedChunk

logger = get_logger(__name__)


def distance_to_score(distance: float | int | None) -> float:
    if distance is None:
        return 0.0
    return max(0.0, min(1.0, 1.0 - float(distance)))


def add_e5_query_prefix(text: str) -> str:
    return f"query: {text.strip()}"


@lru_cache(maxsize=1)
def _get_embedding_model() -> Any:
    try:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model: %s", settings.embedding_model_name)
        return SentenceTransformer(settings.embedding_model_name)
    except Exception as exc:  # pragma: no cover
        raise EmbeddingError("Не удалось загрузить модель эмбеддингов", detail=str(exc)) from exc


@lru_cache(maxsize=1)
def _get_collection() -> Any:
    try:
        import chromadb

        client = chromadb.PersistentClient(path=settings.chroma_persist_directory)
        return client.get_collection(name=settings.chroma_collection_name)
    except ValueError as exc:
        raise RetrievalError(
            "Коллекция ChromaDB не найдена — сначала выполните индексацию "
            "(python -m app.data.indexing.pipeline)",
            detail=str(exc),
        ) from exc
    except Exception as exc:  # pragma: no cover
        raise RetrievalError(
            "Не удалось подключиться к векторной базе",
            detail=str(exc),
        ) from exc


def _resolve_chunk_id(chunk_id: str | None, metadata: dict[str, Any] | None) -> str:
    if chunk_id:
        return str(chunk_id)
    if isinstance(metadata, dict):
        source_path = metadata.get("source_path")
        chunk_index = metadata.get("chunk_index")
        if source_path is not None and chunk_index is not None:
            return f"{source_path}:{chunk_index}"
    return ""


def _build_chunks_from_query_result(result: dict[str, Any]) -> list[RetrievedChunk]:
    ids_row = (result.get("ids") or [[]])[0]
    docs = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    chunks: list[RetrievedChunk] = []
    for idx, (text, metadata, distance) in enumerate(zip(docs, metadatas, distances, strict=False)):
        if not text:
            continue
        meta = dict(metadata) if isinstance(metadata, dict) else {}
        raw_id = ids_row[idx] if idx < len(ids_row) else None
        resolved_id = _resolve_chunk_id(raw_id, meta)
        if not resolved_id:
            continue
        chunks.append(
            RetrievedChunk(
                chunk_id=resolved_id,
                text=str(text),
                metadata=meta,
                score=distance_to_score(distance),
            )
        )
    return chunks


def _embed_query(question: str) -> list[float]:
    try:
        vector = _get_embedding_model().encode(
            add_e5_query_prefix(question),
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vector.tolist()
    except EmbeddingError:
        raise
    except Exception as exc:
        raise EmbeddingError("Не удалось создать эмбеддинг запроса", detail=str(exc)) from exc


def search_context(question: str, top_k: int | None = None) -> list[RetrievedChunk]:
    normalized_question = question.strip()
    if not normalized_question:
        return []

    k = top_k if top_k is not None and top_k > 0 else settings.retrieval_top_k
    logger.debug("Retrieval started: top_k=%s, question_len=%s", k, len(normalized_question))

    query_embedding = _embed_query(normalized_question)
    collection = _get_collection()

    if collection.count() == 0:
        logger.warning(
            "ChromaDB collection %r is empty — run: python -m app.data.indexing.pipeline",
            settings.chroma_collection_name,
        )
        return []

    try:
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        raise RetrievalError("Ошибка поиска в ChromaDB", detail=str(exc)) from exc

    chunks = _build_chunks_from_query_result(result)
    logger.info("Retrieval finished: found_chunks=%s", len(chunks))
    return chunks
