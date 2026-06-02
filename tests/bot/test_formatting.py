"""Unit-тесты для app/bot/formatting.py."""

from __future__ import annotations

import pytest

from app.bot.formatting import (
    TELEGRAM_MAX_MESSAGE_CHARS,
    compose_final_message,
    format_sources,
    split_for_telegram,
)
from app.models.chat import Source


def _source(path: str, score: float = 0.9) -> Source:
    return Source(chunk_id=f"id::{path}", source_path=path, text="...", score=score)


def test_split_returns_empty_list_for_empty_input() -> None:
    assert split_for_telegram("") == []


def test_split_returns_single_part_when_under_limit() -> None:
    text = "короткий текст"
    assert split_for_telegram(text, limit=4096) == [text]


def test_split_prefers_paragraph_boundary() -> None:
    text = "A" * 100 + "\n\n" + "B" * 100
    parts = split_for_telegram(text, limit=150)
    assert len(parts) == 2
    assert parts[0] == "A" * 100
    assert parts[1] == "B" * 100


def test_split_prefers_line_boundary_when_no_paragraph() -> None:
    text = "A" * 100 + "\n" + "B" * 100
    parts = split_for_telegram(text, limit=150)
    assert parts == ["A" * 100, "B" * 100]


def test_split_falls_back_to_space_boundary() -> None:
    text = "слово " * 50
    parts = split_for_telegram(text.strip(), limit=60)
    assert all(len(part) <= 60 for part in parts)
    assert " ".join(parts).replace("  ", " ") == text.strip().replace("  ", " ")


def test_split_handles_text_without_separators() -> None:
    text = "X" * 8200
    parts = split_for_telegram(text, limit=4000)
    assert all(len(part) <= 4000 for part in parts)
    assert "".join(parts) == text


def test_split_default_limit_matches_telegram_constant() -> None:
    text = "Y" * (TELEGRAM_MAX_MESSAGE_CHARS + 5)
    parts = split_for_telegram(text)
    assert all(len(part) <= TELEGRAM_MAX_MESSAGE_CHARS for part in parts)
    assert sum(len(part) for part in parts) == len(text)


def test_split_rejects_non_positive_limit() -> None:
    with pytest.raises(ValueError):
        split_for_telegram("test", limit=0)


def test_format_sources_empty() -> None:
    assert format_sources([]) == ""


def test_format_sources_lists_paths_with_numbering() -> None:
    sources = [_source("instructions/a.txt"), _source("law/b.txt")]
    result = format_sources(sources)
    assert "Источники:" in result
    assert "1. instructions/a.txt" in result
    assert "2. law/b.txt" in result


def test_format_sources_truncates_with_counter() -> None:
    sources = [_source(f"f{i}.txt") for i in range(10)]
    result = format_sources(sources, max_items=3)
    assert "1. f0.txt" in result
    assert "3. f2.txt" in result
    assert "4. f3.txt" not in result
    assert "и ещё 7" in result


def test_compose_final_message_appends_sources_block() -> None:
    answer = "Чтобы зарегистрироваться, заполните форму."
    sources = [_source("instructions/registration.fiz.txt")]
    composed = compose_final_message(answer, sources)
    assert composed.startswith(answer)
    assert composed.endswith("instructions/registration.fiz.txt")


def test_compose_final_message_without_sources_returns_only_answer() -> None:
    assert compose_final_message("Ответ.", []) == "Ответ."
