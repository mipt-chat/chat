from app.rag.retriever import (
    _build_chunks_from_query_result,
    _resolve_chunk_id,
    add_e5_query_prefix,
    distance_to_score,
)


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
