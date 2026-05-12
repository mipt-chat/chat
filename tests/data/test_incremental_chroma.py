"""
Инкрементальная индексация: пустая коллекция, повтор без изменений, правка одного txt.
Эмбеддер подменяется — без сети и без загрузки HF-модели.
"""

from __future__ import annotations

from pathlib import Path

import chromadb
import numpy as np
import pytest

import app.data.indexing.pipeline as pipeline_mod
from app.core.config import Settings
from app.data.indexing.pipeline import (
    compute_document_hash,
    run_indexing_pipeline,
)


class _FakeSentenceTransformer:
    """Возвращает детерминированные векторы фиксированной размерности."""

    dim = 8

    def encode(
        self,
        texts: list[str],
        *,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
    ) -> np.ndarray:
        n = len(texts)
        arr = np.arange(n * self.dim, dtype=np.float32).reshape(n, self.dim)
        if normalize_embeddings:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            arr = arr / norms
        return arr


def _make_kb(tmp_path: Path) -> tuple[Path, Path, Path]:
    kb = tmp_path / "kb"
    kb.mkdir()
    yaml_path = kb / "kb.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "docs:",
                "  doc.a:",
                "    type: markdown",
                "    source: a.txt",
                "  doc.b:",
                "    type: markdown",
                "    source: b.txt",
            ]
        ),
        encoding="utf-8",
    )
    a_txt = kb / "a.txt"
    b_txt = kb / "b.txt"
    a_txt.write_text("doc a version one", encoding="utf-8")
    b_txt.write_text("doc b stable", encoding="utf-8")
    return yaml_path, a_txt, b_txt


def _patch_settings_and_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    kb_yaml: Path,
) -> chromadb.Collection:
    persist = str(tmp_path / "chroma_data")
    s = Settings(
        knowledge_base_file=kb_yaml,
        chroma_persist_directory=persist,
        chroma_collection_name="incremental_mvp_test",
        chunk_size=500,
        chunk_overlap=0,
        embedding_model_name="mock/ignored",
    )
    monkeypatch.setattr(pipeline_mod, "settings", s)
    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer",
        lambda *_a, **_kw: _FakeSentenceTransformer(),
    )
    client = chromadb.PersistentClient(path=persist)
    return client.get_or_create_collection(
        name="incremental_mvp_test",
        metadata={"hnsw:space": "cosine"},
    )


def test_incremental_first_run_fills_empty_collection(tmp_path, monkeypatch) -> None:
    kb_yaml, a_txt, b_txt = _make_kb(tmp_path)
    col = _patch_settings_and_model(monkeypatch, tmp_path, kb_yaml)

    assert col.count() == 0

    run_indexing_pipeline()

    assert col.count() == 2
    data = col.get(include=["metadatas", "documents"])
    ids = data["ids"]
    assert len(ids) == len(set(ids))
    paths = {m["source_path"] for m in data["metadatas"] if m}
    assert str(a_txt.resolve()) in paths
    assert str(b_txt.resolve()) in paths


def test_incremental_second_run_unchanged_no_duplicates(tmp_path, monkeypatch) -> None:
    kb_yaml, _, _ = _make_kb(tmp_path)
    col = _patch_settings_and_model(monkeypatch, tmp_path, kb_yaml)

    run_indexing_pipeline()
    first_ids = set(col.get()["ids"])
    assert col.count() == 2

    run_indexing_pipeline()
    assert col.count() == 2
    second_ids = set(col.get()["ids"])
    assert first_ids == second_ids


def test_incremental_one_file_changed_replaces_chunks_not_duplicates(tmp_path, monkeypatch) -> None:
    kb_yaml, a_txt, b_txt = _make_kb(tmp_path)
    col = _patch_settings_and_model(monkeypatch, tmp_path, kb_yaml)

    run_indexing_pipeline()
    assert col.count() == 2
    path_a = str(a_txt.resolve())
    path_b = str(b_txt.resolve())
    hash_b_before = compute_document_hash("doc b stable")

    a_txt.write_text("doc a version two updated", encoding="utf-8")

    run_indexing_pipeline()

    assert col.count() == 2
    data = col.get(include=["metadatas", "documents"])
    by_source: dict[str, list[tuple[str, str, str]]] = {}
    for cid, meta, doc in zip(
        data["ids"],
        data["metadatas"],
        data["documents"],
        strict=True,
    ):
        sp = meta["source_path"]
        by_source.setdefault(sp, []).append((cid, meta.get("doc_hash", ""), doc))

    assert len(by_source[path_a]) == 1
    assert by_source[path_a][0][2] == "doc a version two updated"
    assert by_source[path_a][0][1] == compute_document_hash("doc a version two updated")

    assert len(by_source[path_b]) == 1
    assert by_source[path_b][0][2] == "doc b stable"
    assert by_source[path_b][0][1] == hash_b_before
