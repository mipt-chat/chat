from collections import Counter
from pathlib import Path

import pytest

from app.core.config import settings
from app.data.indexing.pipeline import (
    _chroma_metadata_for_index,
    add_e5_passage_prefix,
    compute_document_hash,
    load_knowledge_documents,
    select_documents_for_incremental_upsert,
)
from app.models import KnowledgeDocument


def test_chroma_metadata_omits_none_optional_fields() -> None:
    """Chroma не принимает None в metadata — для plain txt в манифесте нет doc_id/yaml_path."""
    doc = KnowledgeDocument(
        source_path="/tmp/knowledge.txt",
        content="hello world",
        metadata={"source_path": "/tmp/knowledge.txt"},
    )
    meta = _chroma_metadata_for_index(doc, 0)
    assert meta["source_path"] == "/tmp/knowledge.txt"
    assert meta["chunk_index"] == 0
    assert "doc_hash" in meta
    assert "doc_id" not in meta
    assert "yaml_path" not in meta


def test_add_e5_passage_prefix() -> None:
    text = "Как вернуть товар?"

    result = add_e5_passage_prefix(text)

    assert result == "passage: Как вернуть товар?"


def test_load_knowledge_documents_from_yaml_import_tree(tmp_path) -> None:
    knowledge_base = tmp_path / "knowledge_base"
    root_dir = knowledge_base / "root"
    instructions_dir = knowledge_base / "instructions"
    terminology_dir = instructions_dir / "terminology"

    root_dir.mkdir(parents=True)
    terminology_dir.mkdir(parents=True)

    root_yaml = "imports:\n  - instructions/instructions.yaml\n"
    (root_dir / "root.yaml").write_text(root_yaml, encoding="utf-8")
    (instructions_dir / "instructions.yaml").write_text(
        "\n".join(
            [
                "docs:",
                "  instructions.terminology:",
                "    location: terminology",
                "    type: markdown",
                "    source: terminology.txt",
            ]
        ),
        encoding="utf-8",
    )
    (terminology_dir / "terminology.txt").write_text("Термины поддержки", encoding="utf-8")

    docs = load_knowledge_documents(root_dir / "root.yaml")

    assert len(docs) == 1
    assert docs[0].metadata["doc_id"] == "instructions.terminology"
    assert docs[0].content == "Термины поддержки"


def test_load_knowledge_documents_nested_import_only_then_docs(tmp_path) -> None:
    """Как law.yaml → documents.yaml: промежуточный yaml только с imports, docs в дочернем."""
    kb = tmp_path / "kb"
    law = kb / "law"
    documents = law / "documents"
    law.mkdir(parents=True)
    documents.mkdir(parents=True)
    (law / "law.yaml").write_text("imports:\n  - documents/documents.yaml\n", encoding="utf-8")
    (documents / "documents.yaml").write_text(
        "\n".join(
            [
                "docs:",
                "  d.one:",
                "    location: sub",
                "    type: markdown",
                "    source: one.txt",
            ]
        ),
        encoding="utf-8",
    )
    (documents / "sub").mkdir()
    (documents / "sub" / "one.txt").write_text("nested body", encoding="utf-8")

    docs = load_knowledge_documents(law / "law.yaml")

    assert len(docs) == 1
    assert docs[0].content == "nested body"
    assert docs[0].metadata["doc_id"] == "d.one"
    assert docs[0].metadata["yaml_path"] == str((documents / "documents.yaml").resolve())


def test_load_knowledge_documents_real_repo_root_yaml_chain() -> None:
    """Репозиторий: root.yaml → imports → … → law/law.yaml → documents/documents.yaml → txt."""
    repo = Path(__file__).resolve().parents[2]
    entry = repo / settings.knowledge_base_file
    if not entry.exists():
        pytest.skip(f"No knowledge base at {entry}")

    docs = load_knowledge_documents(entry)

    nested_manifest = (repo / "knowledge_base/law/documents/documents.yaml").resolve()
    by_yaml = Counter(
        Path(d.metadata["yaml_path"]).resolve()
        for d in docs
        if d.metadata.get("yaml_path")
    )
    assert by_yaml[nested_manifest] == 12
    assert len(docs) == 51
    law_docs = [d for d in docs if Path(d.metadata["yaml_path"]).resolve() == nested_manifest]
    assert len(law_docs) == 12
    assert all("law/documents" in d.source_path.replace("\\", "/") for d in law_docs)


def test_select_documents_for_incremental_upsert_only_changed_docs() -> None:
    doc_unchanged = KnowledgeDocument(source_path="a.txt", content="same", metadata={})
    doc_changed = KnowledgeDocument(source_path="b.txt", content="new", metadata={})
    existing_hashes = {
        "a.txt": compute_document_hash("same"),
        "b.txt": compute_document_hash("old"),
    }

    docs = select_documents_for_incremental_upsert(
        knowledge_docs=[doc_unchanged, doc_changed],
        existing_hash_by_source=existing_hashes,
    )

    assert [doc.source_path for doc in docs] == ["b.txt"]


def test_load_knowledge_documents_resolves_flat_txt_beside_yaml(tmp_path) -> None:
    """YAML задаёт location как подпапку, но файл лежит рядом с yaml (как в knowledge_base/book/)."""
    book_dir = tmp_path / "book"
    book_dir.mkdir()
    (book_dir / "book.yaml").write_text(
        "\n".join(
            [
                "docs:",
                "  books.about_cfa:",
                "    location: about_cfa",
                "    type: markdown",
                "    source: about_cfa.txt",
            ]
        ),
        encoding="utf-8",
    )
    (book_dir / "about_cfa.txt").write_text("CFA текст", encoding="utf-8")

    docs = load_knowledge_documents(book_dir / "book.yaml")

    assert len(docs) == 1
    assert docs[0].content == "CFA текст"
    assert str(docs[0].source_path).endswith("about_cfa.txt")

