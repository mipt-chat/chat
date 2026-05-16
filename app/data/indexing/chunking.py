"""
Утилиты разбиения текста базы знаний на чанки.
"""

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass(frozen=True)
class Chunk:
    """
    Внутренний результат чанкера (не экспортируется наружу из пайплайна).
    Для Chroma и контрактов между слоями используйте ``IndexedChunk`` из ``app.models``.
    """

    text: str
    index: int


def split_text_into_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """
    Разбивает текст на чанки с перекрытием через RecursiveCharacterTextSplitter.

    Сначала по крупным границам (абзацы, строки, конец предложения), затем по пробелу и символам.
    Подходит для последующего эмбеддинга (в т.ч. intfloat/multilingual-e5-base).

    Позиции символов в исходном файле не хранятся: для ответа пользователю и отладки достаточно
    текста чанка и source_path / doc_id в метаданных Chroma.

    Args:
        text: Исходный текст документа.
        chunk_size: Целевой размер чанка в символах.
        chunk_overlap: Перекрытие соседних чанков в символах.

    Returns:
        Список чанков с порядковым index внутри документа.
    """
    normalized = text.strip()
    if not normalized:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size должен быть > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap не может быть < 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap должен быть меньше chunk_size")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    pieces = splitter.split_text(normalized)

    chunks: list[Chunk] = []
    chunk_index = 0
    for piece in pieces:
        body = piece.strip()
        if not body:
            continue
        chunks.append(Chunk(text=body, index=chunk_index))
        chunk_index += 1

    return chunks
