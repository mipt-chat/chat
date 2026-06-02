"""Unit-тесты для app/bot/streaming.py."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter

from app.bot.streaming import StreamingMessageRenderer


@dataclass
class _FakeMessage:
    """Минимальный фейк aiogram.Message: запоминает все edit_text/send_new."""

    message_id: int = 1
    text: str = ""
    edit_calls: list[str] = field(default_factory=list)
    # Если очередь не пуста — поднимаем соответствующее исключение из неё на следующем edit_text
    raise_queue: list[Exception] = field(default_factory=list)

    async def edit_text(self, text: str) -> None:
        if self.raise_queue:
            exc = self.raise_queue.pop(0)
            raise exc
        self.edit_calls.append(text)
        self.text = text


@dataclass
class _Sender:
    """Фейк callback-а send_new — выдаёт новые _FakeMessage с инкрементом message_id."""

    sent: list[_FakeMessage] = field(default_factory=list)
    next_id: int = 100

    async def __call__(self, text: str) -> _FakeMessage:
        msg = _FakeMessage(message_id=self.next_id, text=text)
        self.next_id += 1
        self.sent.append(msg)
        return msg


def _make_bad_request(message: str) -> TelegramBadRequest:
    """Construct TelegramBadRequest across aiogram versions (signature varies)."""
    try:
        return TelegramBadRequest(method=None, message=message)  # type: ignore[arg-type]
    except TypeError:
        return TelegramBadRequest(message=message)  # type: ignore[call-arg]


def _make_retry_after(seconds: float) -> TelegramRetryAfter:
    try:
        return TelegramRetryAfter(method=None, message="flood", retry_after=seconds)  # type: ignore[arg-type]
    except TypeError:
        return TelegramRetryAfter(message="flood", retry_after=seconds)  # type: ignore[call-arg]


@pytest.mark.asyncio
async def test_push_accumulates_tokens_and_edits_after_throttle() -> None:
    draft = _FakeMessage(message_id=1)
    sender = _Sender()
    renderer = StreamingMessageRenderer(
        draft_message=draft,
        send_new=sender,
        throttle_seconds=0.0,  # отключаем throttle для теста
        message_char_limit=4096,
    )

    await renderer.push("При")
    await renderer.push("вет!")

    # Был как минимум один edit, в нём есть весь накопленный текст
    assert draft.edit_calls, "ожидался edit_text"
    assert draft.edit_calls[-1] == "Привет!"


@pytest.mark.asyncio
async def test_finalize_writes_text_via_edit() -> None:
    draft = _FakeMessage(message_id=1)
    sender = _Sender()
    renderer = StreamingMessageRenderer(
        draft_message=draft,
        send_new=sender,
        throttle_seconds=0.0,
        message_char_limit=4096,
    )

    await renderer.finalize("Финальный ответ")

    assert draft.edit_calls[-1] == "Финальный ответ"
    assert sender.sent == []  # длина меньше лимита — новых сообщений не отправляли


@pytest.mark.asyncio
async def test_finalize_splits_long_text_across_messages() -> None:
    draft = _FakeMessage(message_id=1)
    sender = _Sender()
    renderer = StreamingMessageRenderer(
        draft_message=draft,
        send_new=sender,
        throttle_seconds=0.0,
        message_char_limit=100,  # маленький лимит для теста
    )

    long_text = "слово " * 80  # ~480 символов, по 100 = несколько частей
    await renderer.finalize(long_text.strip())

    # Все edit-ы и send_new-ы не должны превышать лимит
    assert all(len(c) <= 100 for c in draft.edit_calls)
    assert all(len(m.text) <= 100 for m in sender.sent)
    # Хотя бы одно новое сообщение должно быть отправлено
    assert sender.sent, "длинный финал должен был породить минимум одно send_new"


@pytest.mark.asyncio
async def test_split_never_exceeds_limit_for_text_without_newlines() -> None:
    """Регрессия на Copilot bug #2: head/tail на сплите не должны превышать лимит."""
    draft = _FakeMessage(message_id=1)
    sender = _Sender()
    limit = 50
    renderer = StreamingMessageRenderer(
        draft_message=draft,
        send_new=sender,
        throttle_seconds=0.0,
        message_char_limit=limit,
    )

    # Один длинный токен без переносов — наивный rpartition("\n") вернул бы целиком в head
    big_token = "x" * 200
    await renderer.push(big_token)

    # Главная проверка: ни один отправленный/отредактированный кусок не больше лимита
    assert all(len(c) <= limit for c in draft.edit_calls), draft.edit_calls
    assert all(len(m.text) <= limit for m in sender.sent), [m.text for m in sender.sent]


@pytest.mark.asyncio
async def test_retry_after_triggers_one_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Регрессия на Copilot bug #3: после RetryAfter рендерер обязан повторить edit."""
    # Подменяем asyncio.sleep, чтобы тест не ждал реальные секунды
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("app.bot.streaming.asyncio.sleep", fake_sleep)

    draft = _FakeMessage(message_id=1)
    draft.raise_queue = [_make_retry_after(2.5)]  # первый edit — RetryAfter, второй — успех

    sender = _Sender()
    renderer = StreamingMessageRenderer(
        draft_message=draft,
        send_new=sender,
        throttle_seconds=0.0,
        message_char_limit=4096,
    )

    await renderer.finalize("Готовый ответ")

    assert sleeps == [2.5]
    assert draft.edit_calls == ["Готовый ответ"], "после ретрая edit должен пройти"


@pytest.mark.asyncio
async def test_bad_request_message_not_modified_is_silently_ignored() -> None:
    draft = _FakeMessage(message_id=1)
    draft.raise_queue = [_make_bad_request("Bad Request: message is not modified")]
    sender = _Sender()
    renderer = StreamingMessageRenderer(
        draft_message=draft,
        send_new=sender,
        throttle_seconds=0.0,
        message_char_limit=4096,
    )

    await renderer.finalize("Тот же текст")

    # Никаких лишних send_new и никаких exceptions
    assert sender.sent == []
    assert draft.edit_calls == []  # edit упал, повторных попыток нет


@pytest.mark.asyncio
async def test_bad_request_other_falls_back_to_send_new() -> None:
    draft = _FakeMessage(message_id=1)
    draft.raise_queue = [_make_bad_request("Bad Request: chat not found")]
    sender = _Sender()
    renderer = StreamingMessageRenderer(
        draft_message=draft,
        send_new=sender,
        throttle_seconds=0.0,
        message_char_limit=4096,
    )

    await renderer.finalize("Текст")

    # Должен прибегнуть к send_new вместо edit
    assert len(sender.sent) == 1
    assert sender.sent[0].text == "Текст"


@pytest.mark.asyncio
async def test_finalize_is_idempotent() -> None:
    draft = _FakeMessage(message_id=1)
    sender = _Sender()
    renderer = StreamingMessageRenderer(
        draft_message=draft,
        send_new=sender,
        throttle_seconds=0.0,
        message_char_limit=4096,
    )

    await renderer.finalize("Первый финал")
    await renderer.finalize("Второй финал")  # должен быть проигнорирован

    assert draft.edit_calls == ["Первый финал"]


@pytest.mark.asyncio
async def test_push_is_noop_after_finalize() -> None:
    draft = _FakeMessage(message_id=1)
    sender = _Sender()
    renderer = StreamingMessageRenderer(
        draft_message=draft,
        send_new=sender,
        throttle_seconds=0.0,
        message_char_limit=4096,
    )

    await renderer.finalize("Готово")
    await renderer.push("ещё токен")

    # push после finalize ничего не делает
    assert draft.edit_calls == ["Готово"]
    assert sender.sent == []
