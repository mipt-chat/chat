from unittest.mock import MagicMock, patch

import pytest

from app.core.config import settings
from app.core.exceptions import RetrievalError
from app.rag.retriever import (
    _build_chunks_from_query_result,
    _resolve_chunk_id,
    add_e5_query_prefix,
    distance_to_score,
    search_context,
)


class _FakeCollection:
    def __init__(
        self,
        *,
        count: int = 1,
        query_result: dict | None = None,
        query_error: Exception | None = None,
    ) -> None:
        self._count = count
        self._query_result = query_result or {
            "ids": [["kb/doc.txt:0"]],
            "documents": [["ответ из базы"]],
            "metadatas": [[{"source_path": "kb/doc.txt", "chunk_index": 0}]],
            "distances": [[0.15]],
        }
        self._query_error = query_error
        self.last_query_kwargs: dict | None = None

    def count(self) -> int:
        return self._count

    def query(self, **kwargs):
        self.last_query_kwargs = kwargs
        if self._query_error is not None:
            raise self._query_error
        return self._query_result


def test_add_e5_query_prefix() -> None:
    assert add_e5_query_prefix("Как вернуть товар?") == "query: Как вернуть товар?"


def test_distance_to_score_cosine() -> None:
    assert distance_to_score(0.0) == 1.0
    assert distance_to_score(1.0) == 0.0
    assert distance_to_score(None) == 0.0
    assert 0.0 <= distance_to_score(0.3) <= 1.0


def test_resolve_chunk_id_from_metadata() -> None:
    assert _resolve_chunk_id(None, {"source_path": "/kb/a.txt", "chunk_index": 2}) == "/kb/a.txt:2"


def test_build_chunks_from_query_result() -> None:
    result = {
        "ids": [["path/doc.txt:0", "path/doc.txt:1"]],
        "documents": [["text one", "text two"]],
        "metadatas": [
            [
                {"source_path": "/kb/law/doc.txt", "chunk_index": 0},
                {"source_path": "/kb/instructions/doc.txt", "chunk_index": 1},
            ]
        ],
        "distances": [[0.1, 0.5]],
    }

    chunks = _build_chunks_from_query_result(result)

    assert len(chunks) == 2
    assert chunks[0].chunk_id == "path/doc.txt:0"
    assert chunks[0].metadata["source_path"] == "/kb/law/doc.txt"
    assert chunks[0].score > chunks[1].score


def test_build_chunks_without_ids_in_result() -> None:
    result = {
        "documents": [["only text"]],
        "metadatas": [[{"source_path": "/kb/doc.txt", "chunk_index": 0}]],
        "distances": [[0.2]],
    }
    chunks = _build_chunks_from_query_result(result)
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "/kb/doc.txt:0"


@patch("app.rag.retriever._get_collection")
@patch("app.rag.retriever._embed_query", return_value=[0.1, 0.2, 0.3])
def test_search_context_success(mock_embed: MagicMock, mock_get_collection: MagicMock) -> None:
    fake = _FakeCollection()
    mock_get_collection.return_value = fake

    chunks = search_context("  вопрос про возврат  ", top_k=2)

    mock_embed.assert_called_once_with("вопрос про возврат")
    assert fake.last_query_kwargs is not None
    assert fake.last_query_kwargs["query_embeddings"] == [[0.1, 0.2, 0.3]]
    assert fake.last_query_kwargs["n_results"] == 2
    assert fake.last_query_kwargs["include"] == ["documents", "metadatas", "distances"]
    assert len(chunks) == 1
    assert chunks[0].text == "ответ из базы"
    assert chunks[0].chunk_id == "kb/doc.txt:0"
    assert chunks[0].score == pytest.approx(0.85)


@patch("app.rag.retriever._get_collection")
@patch("app.rag.retriever._embed_query")
def test_search_context_empty_question_skips_chroma(
    mock_embed: MagicMock,
    mock_get_collection: MagicMock,
) -> None:
    assert search_context("   ") == []
    mock_embed.assert_not_called()
    mock_get_collection.assert_not_called()


@patch("app.rag.retriever._get_collection")
@patch("app.rag.retriever._embed_query", return_value=[0.0])
def test_search_context_empty_collection_returns_empty(
    mock_embed: MagicMock,
    mock_get_collection: MagicMock,
) -> None:
    fake = _FakeCollection(count=0)
    mock_get_collection.return_value = fake

    assert search_context("вопрос") == []
    mock_embed.assert_called_once()
    assert fake.last_query_kwargs is None


@patch("app.rag.retriever._get_collection")
@patch("app.rag.retriever._embed_query", return_value=[0.0])
def test_search_context_query_failure_raises_retrieval_error(
    mock_embed: MagicMock,
    mock_get_collection: MagicMock,
) -> None:
    fake = _FakeCollection(query_error=RuntimeError("chroma down"))
    mock_get_collection.return_value = fake

    with pytest.raises(RetrievalError, match="Ошибка поиска в ChromaDB"):
        search_context("вопрос")

    mock_embed.assert_called_once()


@patch("app.rag.retriever._get_collection")
@patch("app.rag.retriever._embed_query", return_value=[0.0])
def test_search_context_default_top_k_from_settings(
    mock_embed: MagicMock,
    mock_get_collection: MagicMock,
) -> None:
    fake = _FakeCollection()
    mock_get_collection.return_value = fake

    search_context("вопрос", top_k=None)

    assert fake.last_query_kwargs is not None
    assert fake.last_query_kwargs["n_results"] == settings.retrieval_top_k
