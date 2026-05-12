from app.data.indexing.pipeline import add_e5_passage_prefix
from app.data.indexing.pipeline import compute_document_hash
from app.data.indexing.pipeline import load_knowledge_documents
from app.data.indexing.pipeline import select_documents_for_incremental_upsert
from app.models import KnowledgeDocument


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

    (root_dir / "root.yaml").write_text("imports:\n  - instructions/instructions.yaml\n", encoding="utf-8")
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

