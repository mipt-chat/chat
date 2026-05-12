"""
Запись проиндексированных чанков в ChromaDB.
На вход принимает только общие Pydantic-модели (IndexedChunk).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.models import IndexedChunk

if TYPE_CHECKING:
    import chromadb


def add_indexed_chunks(
    collection: chromadb.Collection,
    chunks: list[IndexedChunk],
    embeddings: list[list[float]],
) -> None:
    """
    Добавляет векторы и метаданные в коллекцию Chroma.

    Args:
        collection: Коллекция ChromaDB.
        chunks: Чанки с уже собранными ``chunk_id``, ``text``, ``metadata``.
        embeddings: Векторы в том же порядке, что и ``chunks``.
    """
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"chunks and embeddings length mismatch: {len(chunks)} vs {len(embeddings)}",
        )
    collection.add(
        ids=[c.chunk_id for c in chunks],
        embeddings=embeddings,
        documents=[c.text for c in chunks],
        metadatas=[dict(c.metadata) for c in chunks],
    )
