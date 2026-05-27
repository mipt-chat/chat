"""
Unit-тесты для app/bot/handlers.py.

Эмулируем минимальную часть aiogram (Message, Bot, Chat, User), достаточную
для проверки логики хендлеров без поднятия настоящего Bot/HTTP.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import pytest

from app.bot import handlers as bot_handlers
from app.bot.api_client import BackendAPIError, StreamEvent
from app.bot.handlers import (
    ERROR_REPLY,
    NON_TEXT_REPLY,
    _handle_user_question,
)
from app.bot.session import SessionRegistry
from app.models.chat import ChatResponse, Source


@dataclass
class _FakeUser:
    id: int = 100


@dataclass
class _FakeChat:
    id: int = 555
    type: str = "private"


@dataclass
class _SentMessage:
    text: str
    message_id: int

    async def edit_text(self, text: str) -> None:
        self.text = text


@dataclass
class _FakeBot:
    chat_actions: list[tuple[int, str]] = field(default_factory=list)

    async def send_chat_action(self, chat_id: int, action: str) -> None:
        self.chat_actions.append((chat_id, action))


@dataclass
class _FakeMessage:
    text: str | None = "вопрос"
    chat: _FakeChat = field(default_factory=_FakeChat)
    from_user: _FakeUser | None = field(default_factory=_FakeUser)
    bot: _FakeBot = field(default_factory=_FakeBot)
    answered: list[_SentMessage] = field(default_factory=list)
    _next_id: int = 1000

    async def answer(self, text: str) -> _SentMessage:
        msg = _SentMessage(text=text, message_id=self._next_id)
        self._next_id += 1
        self.answered.append(msg)
        return msg


class _FakeAPIClient:
    """Заглушка BackendAPIClient для тестов хендлеров."""

    def __init__(
        self,
        *,
        chat_response: ChatResponse | None = None,
        stream_events: list[StreamEvent] | None = None,
        raise_on_chat: BackendAPIError | None = None,
        raise_on_stream: BackendAPIError | None = None,
    ) -> None:
        self.chat_response = chat_response
        self.stream_events = stream_events or []
        self.raise_on_chat = raise_on_chat
        self.raise_on_stream = raise_on_stream
        self.chat_calls: list[tuple[str, str | None]] = []
        self.stream_calls: list[tuple[str, str | None]] = []

    async def chat(self, message: str, session_id: str | None) -> ChatResponse:
        self.chat_calls.append((message, session_id))
        if self.raise_on_chat is not None:
            raise self.raise_on_chat
        assert self.chat_response is not None
        return self.chat_response

    async def stream_chat(
        self, message: str, session_id: str | None
    ) -> AsyncIterator[StreamEvent]:
        self.stream_calls.append((message, session_id))
        if self.raise_on_stream is not None:
            raise self.raise_on_stream
        for event in self.stream_events:
            yield event


def _source(path: str) -> Source:
    return Source(chunk_id=f"id::{path}", source_path=path, text="...", score=0.9)


def _force_no_streaming(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bot_handlers.settings, "bot_streaming_enabled", False)


def _force_streaming(monkeypatch: pytest.MonkeyPatch, *, throttle: float = 0.0) -> None:
    monkeypatch.setattr(bot_handlers.settings, "bot_streaming_enabled", True)
    monkeypatch.setattr(bot_handlers.settings, "bot_draft_throttle_seconds", throttle)
    monkeypatch.setattr(bot_handlers.settings, "bot_max_final_message_chars", 4096)


@pytest.mark.asyncio
async def test_plain_reply_calls_api_and_sends_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_no_streaming(monkeypatch)

    api = _FakeAPIClient(
        chat_response=ChatResponse(
            answer="Ответ из API",
            sources=[_source("instructions/a.txt")],
            answered=True,
            session_id="tg_user_100_confirmed",
        )
    )
    sessions = SessionRegistry()
    message = _FakeMessage(text="как зарегистрироваться?")

    await _handle_user_question(message, api, sessions)

    assert len(api.chat_calls) == 1
    assert api.chat_calls[0][0] == "как зарегистрироваться?"
    assert message.answered[0].text.startswith("Ответ из API")
    assert "instructions/a.txt" in message.answered[0].text
    assert sessions.get_or_create(100) == "tg_user_100_confirmed"
    assert message.bot.chat_actions == [(555, "typing")]


@pytest.mark.asyncio
async def test_plain_reply_hides_sources_when_not_answered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_no_streaming(monkeypatch)

    api = _FakeAPIClient(
        chat_response=ChatResponse(
            answer="Извините, не нашёл.",
            sources=[_source("instructions/a.txt")],
            answered=False,
            session_id="tg_user_100_x",
        )
    )
    sessions = SessionRegistry()
    message = _FakeMessage(text="вопрос вне базы")

    await _handle_user_question(message, api, sessions)

    assert message.answered[0].text == "Извините, не нашёл."


@pytest.mark.asyncio
async def test_plain_reply_handles_backend_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_no_streaming(monkeypatch)

    api = _FakeAPIClient(raise_on_chat=BackendAPIError("LLM down", status_code=502))
    sessions = SessionRegistry()
    message = _FakeMessage(text="вопрос")

    await _handle_user_question(message, api, sessions)

    assert message.answered == [_SentMessage(text=ERROR_REPLY, message_id=1000)]


@pytest.mark.asyncio
async def test_plain_reply_splits_long_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_no_streaming(monkeypatch)

    long_answer = ("слово " * 1500).strip()
    api = _FakeAPIClient(
        chat_response=ChatResponse(
            answer=long_answer,
            sources=[],
            answered=True,
            session_id="tg_user_100_z",
        )
    )
    sessions = SessionRegistry()
    message = _FakeMessage(text="дай длинный ответ")

    await _handle_user_question(message, api, sessions)

    assert len(message.answered) >= 2
    assert all(len(m.text) <= 4096 for m in message.answered)


@pytest.mark.asyncio
async def test_group_chat_uses_plain_path_even_when_streaming_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_streaming(monkeypatch)

    api = _FakeAPIClient(
        chat_response=ChatResponse(
            answer="Ответ для группы",
            sources=[],
            answered=True,
            session_id="tg_user_100_g",
        )
    )
    sessions = SessionRegistry()
    message = _FakeMessage(text="привет", chat=_FakeChat(id=42, type="group"))

    await _handle_user_question(message, api, sessions)

    assert len(api.chat_calls) == 1
    assert len(api.stream_calls) == 0
    assert message.answered[0].text == "Ответ для группы"


@pytest.mark.asyncio
async def test_streaming_reply_accumulates_tokens_and_finalizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_streaming(monkeypatch)

    events = [
        StreamEvent(type="session", data={"session_id": "tg_user_100_stream"}),
        StreamEvent(type="token", data={"text": "При"}),
        StreamEvent(type="token", data={"text": "вет!"}),
        StreamEvent(
            type="done",
            data={
                "answer": "Привет!",
                "answered": True,
                "session_id": "tg_user_100_stream",
                "sources": [
                    {
                        "chunk_id": "c1",
                        "source_path": "instructions/hello.txt",
                        "text": "...",
                        "score": 0.95,
                    }
                ],
            },
        ),
    ]
    api = _FakeAPIClient(stream_events=events)
    sessions = SessionRegistry()
    message = _FakeMessage(text="скажи привет")

    await _handle_user_question(message, api, sessions)

    final = message.answered[-1].text
    assert final.startswith("Привет!")
    assert "instructions/hello.txt" in final
    assert sessions.get_or_create(100) == "tg_user_100_stream"


@pytest.mark.asyncio
async def test_streaming_reply_falls_back_on_error_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_streaming(monkeypatch)

    events = [
        StreamEvent(type="session", data={"session_id": "tg_user_100_err"}),
        StreamEvent(type="token", data={"text": "Hello"}),
        StreamEvent(type="error", data={"error": "LLM exploded"}),
    ]
    api = _FakeAPIClient(stream_events=events)
    sessions = SessionRegistry()
    message = _FakeMessage(text="ломаем")

    await _handle_user_question(message, api, sessions)

    last = message.answered[-1].text
    assert last == ERROR_REPLY


def test_build_router_registers_expected_handlers() -> None:
    api = _FakeAPIClient(
        chat_response=ChatResponse(answer="x", sources=[], answered=True, session_id="s")
    )
    sessions = SessionRegistry()
    router = bot_handlers.build_router(api, sessions)
    assert router.name == "customer-support-bot"
    assert len(router.message.handlers) >= 4


def test_non_text_reply_constant_is_user_friendly() -> None:
    assert "текст" in NON_TEXT_REPLY.lower()
