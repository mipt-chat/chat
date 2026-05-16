from app.data.indexing.chunking import split_text_into_chunks


def test_split_text_into_chunks_keeps_overlap() -> None:
    text = "0123456789"

    chunks = split_text_into_chunks(text=text, chunk_size=6, chunk_overlap=2)

    assert len(chunks) == 2
    assert chunks[0].text == "012345"
    assert chunks[0].index == 0
    assert chunks[1].text == "456789"
    assert chunks[1].index == 1


def test_split_text_into_chunks_validates_overlap() -> None:
    text = "test"

    try:
        split_text_into_chunks(text=text, chunk_size=4, chunk_overlap=4)
    except ValueError as exc:
        assert "должен быть меньше chunk_size" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid overlap")

