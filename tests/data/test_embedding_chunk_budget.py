"""
Проверка, что чанки после нарезки укладываются в окно эмбеддинг-модели (токены).

CHUNK_SIZE в Settings задан в символах; у encoder-моделей (E5) лимит — в токенах.
Если чанк + префикс passage: длиннее max_seq_length, модель обрежет вход и качество RAG просядет.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import settings
from app.data.indexing.chunking import Chunk, split_text_into_chunks
from app.data.indexing.pipeline import add_e5_passage_prefix, load_knowledge_documents

pytestmark = pytest.mark.slow


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_indexed_chunks_fit_embedding_model_max_length() -> None:
    pytest.importorskip("sentence_transformers")
    from sentence_transformers import SentenceTransformer

    kb = _project_root() / settings.knowledge_base_file
    if not kb.exists():
        pytest.skip(f"Нет базы знаний по пути {kb}")

    docs = load_knowledge_documents(kb)
    all_chunks: list[Chunk] = []
    for doc in docs:
        all_chunks.extend(
            split_text_into_chunks(
                text=doc.content,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
        )
    assert all_chunks, "Нет чанков после нарезки документов"

    model = SentenceTransformer(settings.embedding_model_name)
    max_seq = int(model.max_seq_length)
    # Небольшой запас под особые токены (CLS/SEP и т.д.) — у ST обычно уже учтено в encode
    assert max_seq > 0

    worst_n = 0
    worst_preview = ""
    for chunk in all_chunks:
        prefixed = add_e5_passage_prefix(chunk.text)
        n = len(model.tokenizer.encode(prefixed, add_special_tokens=True))
        if n > worst_n:
            worst_n = n
            worst_preview = prefixed[:120].replace("\n", " ")

    assert worst_n <= max_seq, (
        f"Самый длинный чанк с префиксом passage: занимает {worst_n} токенов, "
        f"лимит модели max_seq_length={max_seq}. Уменьшите CHUNK_SIZE или overlap. "
        f"Пример начала: {worst_preview!r}..."
    )


def test_worst_case_char_chunk_token_count_against_model() -> None:
    """
    Синтетика: строка ровно из chunk_size символов (плотный текст) — верхняя граница по символам.
    """
    pytest.importorskip("sentence_transformers")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(settings.embedding_model_name)
    max_seq = int(model.max_seq_length)

    dense = "а" * settings.chunk_size
    chunks = split_text_into_chunks(
        text=dense,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    assert len(chunks) >= 1
    prefixed = add_e5_passage_prefix(chunks[0].text)
    n = len(model.tokenizer.encode(prefixed, add_special_tokens=True))

    assert n <= max_seq, (
        f"Даже один чанк размером CHUNK_SIZE={settings.chunk_size} символов даёт "
        f"{n} токенов > max_seq_length={max_seq}. Уменьшите CHUNK_SIZE в Settings / .env."
    )
