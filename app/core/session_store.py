"""
Хранение истории диалогов по session_id.

MVP-реализация на JSON-файлах.
В будущем — замена на PostgreSQL через тот же интерфейс.
"""

import json
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path

from app.core.config import settings
from app.core.logging_config import get_logger
from app.models.session import DialogMessage, Session

logger = get_logger(__name__)


# ============================================================
# Абстрактный интерфейс хранилища сессий
# ============================================================

class BaseSessionStore(ABC):
    """Абстрактный класс хранилища сессий.

    Позволяет заменить JSON на PostgreSQL без изменения
    остального кода — достаточно реализовать этот интерфейс.
    """

    @abstractmethod
    def load_session(self, session_id: str) -> Session:
        """Загружает сессию или создаёт новую."""
        ...

    @abstractmethod
    def save_session(self, session: Session) -> None:
        """Сохраняет сессию."""
        ...

    @abstractmethod
    def cleanup_old_sessions(self) -> int:
        """Удаляет сессии старше TTL. Возвращает количество удалённых."""
        ...


# ============================================================
# JSON-реализация (MVP)
# ============================================================

class JsonSessionStore(BaseSessionStore):
    """Хранилище сессий в JSON-файлах.

    Каждая сессия — отдельный файл в data/sessions/.
    """

    def __init__(self, sessions_dir: Path | None = None) -> None:
        self._sessions_dir = sessions_dir or Path("data/sessions")
        self._sessions_dir.mkdir(parents=True, exist_ok=True)

    def _get_session_path(self, session_id: str) -> Path:
        """Возвращает путь к файлу сессии."""
        safe_id = "".join(c for c in session_id if c.isalnum() or c in "_-")
        return self._sessions_dir / f"{safe_id}.json"

    def load_session(self, session_id: str) -> Session:
        file_path = self._get_session_path(session_id)

        if not file_path.exists():
            logger.debug(f"Creating new session: {session_id}")
            return Session(session_id=session_id, messages=[], updated_at=datetime.now())

        try:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
            session = Session.model_validate(data)
            logger.debug(f"Loaded session {session_id} with {len(session.messages)} messages")
            return session
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Corrupted session file for {session_id}, creating new: {e}")
            return Session(session_id=session_id, messages=[], updated_at=datetime.now())

    def save_session(self, session: Session) -> None:
        file_path = self._get_session_path(session.session_id)
        session.updated_at = datetime.now()

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(session.model_dump_json(indent=2, ensure_ascii=False))
            logger.debug(f"Saved session {session.session_id} with {len(session.messages)} messages")
        except OSError as e:
            logger.error(f"Failed to save session {session.session_id}: {e}")

    def cleanup_old_sessions(self) -> int:
        if settings.session_ttl_days <= 0:
            logger.debug("Session TTL is 0, skipping cleanup")
            return 0

        if not self._sessions_dir.exists():
            return 0

        cutoff = datetime.now() - timedelta(days=settings.session_ttl_days)
        deleted = 0

        for file_path in self._sessions_dir.glob("*.json"):
            try:
                mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                if mtime < cutoff:
                    file_path.unlink()
                    deleted += 1
                    logger.debug(f"Deleted expired session: {file_path.name}")
            except OSError as e:
                logger.warning(f"Failed to delete session {file_path.name}: {e}")

        if deleted > 0:
            logger.info(f"Cleaned up {deleted} expired sessions")
        return deleted


# ============================================================
# Глобальный экземпляр хранилища
# ============================================================

# В MVP — JSON. После MVP меняем на PostgresSessionStore
session_store: BaseSessionStore = JsonSessionStore()


# ============================================================
# Публичные функции (фасад)
# ============================================================

def load_session(session_id: str) -> Session:
    """Загружает сессию из хранилища или создаёт новую."""
    return session_store.load_session(session_id)


def save_session(session: Session) -> None:
    """Сохраняет сессию в хранилище."""
    session_store.save_session(session)


def add_message(session_id: str, role: str, content: str) -> Session:
    """Добавляет сообщение в сессию и обрезает историю до лимита."""
    session = load_session(session_id)

    message = DialogMessage(
        message_id=str(uuid.uuid4()),
        role=role,
        content=content,
        timestamp=datetime.now(),
    )
    session.messages.append(message)

    # Обрезаем историю до последних N * 2 сообщений
    max_messages = settings.max_history_length * 2
    if len(session.messages) > max_messages:
        session.messages = session.messages[-max_messages:]

    save_session(session)
    return session


def get_recent_messages(session_id: str) -> list[DialogMessage]:
    """Возвращает последние сообщения сессии для контекста LLM."""
    session = load_session(session_id)
    return session.messages[-settings.max_history_length * 2:]


def cleanup_old_sessions() -> int:
    """Удаляет сессии старше SESSION_TTL_DAYS."""
    return session_store.cleanup_old_sessions()
