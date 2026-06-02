"""
Реестр session_id для пользователей Telegram.

Бэкенд хранит историю диалога по `session_id` (см. app/core/session_store.py).
Бот должен для каждого пользователя передавать стабильный session_id, а при
команде /reset — выдать новый, чтобы бэкенд завёл свежую сессию.

Хранилище — in-memory: при перезапуске бота сессии «обнуляются», и бэкенд
просто создаст их заново на следующем сообщении. Это допустимо для MVP.
"""

from __future__ import annotations

import time
import uuid


def _build_session_id(user_id: int) -> str:
    return f"tg_user_{user_id}_{int(time.time())}_{uuid.uuid4().hex[:6]}"


class SessionRegistry:
    """Маппинг telegram user_id → session_id для бэкенда."""

    def __init__(self) -> None:
        self._sessions: dict[int, str] = {}

    def get_or_create(self, user_id: int) -> str:
        session_id = self._sessions.get(user_id)
        if session_id is None:
            session_id = _build_session_id(user_id)
            self._sessions[user_id] = session_id
        return session_id

    def reset(self, user_id: int) -> str:
        """Сбрасывает сессию пользователя и возвращает новый session_id."""
        session_id = _build_session_id(user_id)
        self._sessions[user_id] = session_id
        return session_id

    def update(self, user_id: int, session_id: str) -> None:
        """Принять session_id, подтверждённый бэкендом (handshake)."""
        if session_id:
            self._sessions[user_id] = session_id
