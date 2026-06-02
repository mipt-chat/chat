"""
Форматирование текста для отправки в Telegram.

Главные ограничения Telegram:
- одно сообщение ≤ 4096 символов;
- forbidden-символы для разметки лучше не использовать —
  отправляем plain text, чтобы LLM-ответ не сломал парсер.
"""

from __future__ import annotations

from app.models.chat import Source

TELEGRAM_MAX_MESSAGE_CHARS: int = 4096


def split_for_telegram(text: str, limit: int = TELEGRAM_MAX_MESSAGE_CHARS) -> list[str]:
    """
    Разбивает длинный текст на части ≤ limit символов.

    Стратегия: предпочитаем границу абзаца (`\\n\\n`), затем — границу строки
    (`\\n`), затем — границу слова (пробел). Последний fallback — жёсткий разрез.
    Пустых частей не возвращаем.
    """
    if limit <= 0:
        raise ValueError("limit must be positive")

    text = text or ""
    if len(text) <= limit:
        return [text] if text else []

    parts: list[str] = []
    remainder = text
    while len(remainder) > limit:
        window = remainder[:limit]
        split_at = _best_split_index(window)
        chunk = remainder[:split_at].rstrip()
        if chunk:
            parts.append(chunk)
        remainder = remainder[split_at:].lstrip()
        if not remainder:
            break

    if remainder:
        parts.append(remainder)
    return parts


def _best_split_index(window: str) -> int:
    """Ищет лучший индекс для разреза в окне ≤ limit."""
    for separator in ("\n\n", "\n", " "):
        idx = window.rfind(separator)
        if idx > 0:
            return idx + len(separator)
    return len(window)


def format_sources(sources: list[Source], max_items: int = 5) -> str:
    """
    Превращает список источников в компактный человекочитаемый текст.
    Возвращает пустую строку, если источников нет.
    """
    if not sources:
        return ""

    lines = ["", "Источники:"]
    for index, source in enumerate(sources[:max_items], start=1):
        path = source.source_path or source.chunk_id
        lines.append(f"{index}. {path}")

    if len(sources) > max_items:
        lines.append(f"...и ещё {len(sources) - max_items}")

    return "\n".join(lines)


def compose_final_message(answer: str, sources: list[Source]) -> str:
    """Склеивает ответ LLM и блок источников."""
    return (answer or "").rstrip() + format_sources(sources)
