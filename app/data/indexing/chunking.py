"""
Утилиты разбиения текста базы знаний на чанки.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """Локальная модель чанка для слоя data processing."""

    text: str
    index: int
    start_char: int
    end_char: int


def split_text_into_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """
    Разбивает текст на последовательные чанки с overlap.

    Args:
        text: Исходный текст документа.
        chunk_size: Размер чанка в символах.
        chunk_overlap: Перекрытие соседних чанков в символах.

    Returns:
        Список чанков с диапазонами символов.
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

    chunks: list[Chunk] = []
    start = 0
    chunk_index = 0
    step = chunk_size - chunk_overlap

    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        chunk_text = normalized[start:end].strip()
        if chunk_text:
            chunks.append(
                Chunk(
                    text=chunk_text,
                    index=chunk_index,
                    start_char=start,
                    end_char=end,
                )
            )
            chunk_index += 1
        if end >= len(normalized):
            break
        start += step

    return chunks

