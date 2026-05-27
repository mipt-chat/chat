"""
«Черновой» стриминг ответа в Telegram через edit_message.

Идея: пока бэкенд по SSE присылает токены, бот накапливает их и обновляет
последнее сообщение в чате не чаще, чем раз в throttle-секунд. Когда буфер
не помещается в одно сообщение Telegram (4096 символов), текущее сообщение
финализируется, а дальнейшие токены идут в новое.

Главное правило: для приватных чатов draft-стриминг улучшает UX, но в группах
он бесполезен и шумит. Решение «использовать стриминг или нет» принимается
в хендлере выше; этот модуль просто умеет рисовать стрим.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from time import monotonic

from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter
from aiogram.types import Message

from app.bot.formatting import TELEGRAM_MAX_MESSAGE_CHARS, split_for_telegram
from app.core import get_logger

logger = get_logger(__name__)


SendCallable = Callable[[str], Awaitable[Message]]


class StreamingMessageRenderer:
    """
    Постепенно отрисовывает накапливающийся текст в Telegram.

    Поддерживает разбиение длинных ответов на несколько сообщений и
    финальную замену чернового текста на готовый (с источниками).
    """

    def __init__(
        self,
        *,
        draft_message: Message,
        send_new: SendCallable,
        throttle_seconds: float,
        message_char_limit: int = TELEGRAM_MAX_MESSAGE_CHARS,
    ) -> None:
        self._current = draft_message
        self._send_new = send_new
        self._throttle = max(0.0, throttle_seconds)
        self._limit = max(1, message_char_limit)

        self._buffer: list[str] = []
        self._last_rendered: str = ""
        self._last_edit_at: float = 0.0
        self._sent_message_ids: list[int] = [draft_message.message_id]
        self._finalized: bool = False

    async def push(self, token: str) -> None:
        """Добавляет очередной токен и при необходимости обновляет сообщение."""
        if not token or self._finalized:
            return

        self._buffer.append(token)
        text = "".join(self._buffer)

        if len(text) > self._limit:
            await self._flush_current_and_split(text)
            return

        if monotonic() - self._last_edit_at >= self._throttle:
            await self._safe_edit(text)
            self._last_edit_at = monotonic()

    async def finalize(self, final_text: str) -> None:
        """
        Записывает финальный текст (ответ + источники) в чат.

        Если финальный текст влезает — заменяет текущее черновое сообщение,
        иначе дополнительно отправляет хвост новыми сообщениями.
        """
        if self._finalized:
            return
        self._finalized = True

        parts = split_for_telegram(final_text, limit=self._limit) or [""]
        head, *tail = parts

        await self._safe_edit(head)
        for part in tail:
            new_msg = await self._send_new(part)
            self._sent_message_ids.append(new_msg.message_id)

    async def _flush_current_and_split(self, accumulated: str) -> None:
        head, _, tail = accumulated.rpartition("\n")
        if not head:
            head = accumulated[: self._limit]
            tail = accumulated[self._limit :]

        await self._safe_edit(head)

        new_draft = await self._send_new(tail or "…")
        self._current = new_draft
        self._sent_message_ids.append(new_draft.message_id)
        self._buffer = [tail]
        self._last_rendered = ""
        self._last_edit_at = monotonic()

    async def _safe_edit(self, text: str) -> None:
        """Редактирует текущее сообщение, аккуратно обрабатывая лимиты Telegram."""
        if not text:
            return
        if text == self._last_rendered:
            return

        try:
            await self._current.edit_text(text)
            self._last_rendered = text
        except TelegramRetryAfter as exc:
            logger.warning("Telegram rate limit on edit, sleep=%.2fs", exc.retry_after)
            await asyncio.sleep(exc.retry_after)
        except TelegramBadRequest as exc:
            if "message is not modified" in str(exc).lower():
                return
            logger.debug("edit_text failed (%s), falling back to send_new", exc)
            new_msg = await self._send_new(text)
            self._current = new_msg
            self._sent_message_ids.append(new_msg.message_id)
            self._last_rendered = text
