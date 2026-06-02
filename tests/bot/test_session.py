"""Unit-тесты для app/bot/session.py."""

from __future__ import annotations

from app.bot.session import SessionRegistry


def test_get_or_create_returns_stable_session_per_user() -> None:
    registry = SessionRegistry()
    first = registry.get_or_create(42)
    second = registry.get_or_create(42)
    assert first == second
    assert first.startswith("tg_user_42_")


def test_different_users_get_different_sessions() -> None:
    registry = SessionRegistry()
    a = registry.get_or_create(1)
    b = registry.get_or_create(2)
    assert a != b
    assert a.startswith("tg_user_1_")
    assert b.startswith("tg_user_2_")


def test_reset_replaces_session_id() -> None:
    registry = SessionRegistry()
    original = registry.get_or_create(7)
    refreshed = registry.reset(7)
    assert refreshed != original
    assert registry.get_or_create(7) == refreshed


def test_update_accepts_backend_session_id() -> None:
    registry = SessionRegistry()
    registry.get_or_create(99)
    registry.update(99, "tg_user_99_from_backend")
    assert registry.get_or_create(99) == "tg_user_99_from_backend"


def test_update_ignores_empty_session_id() -> None:
    registry = SessionRegistry()
    original = registry.get_or_create(5)
    registry.update(5, "")
    assert registry.get_or_create(5) == original
