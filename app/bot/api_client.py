"""
Асинхронный клиент к REST API бэкенда.

Эндпоинты бэкенда (см. app/api/routes/chat.py):
- POST /chat          → ChatResponse
- POST /chat/stream   → SSE с событиями session/token/sources/done/error

Бот общается с бэкендом только через эти два метода. Никаких знаний про RAG,
LLM, базу знаний и т.д. в боте нет.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx

from app.core import get_logger
from app.models.chat import ChatRequest, ChatResponse, Source

logger = get_logger(__name__)


class BackendAPIError(Exception):
    """Любая проблема при обращении к бэкенду (5xx, 4xx, таймаут, разрыв)."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class StreamEvent:
    """Событие потокового ответа от /chat/stream."""

    type: str
    data: dict[str, Any]


class BackendAPIClient:
    """
    Тонкая обёртка над httpx.AsyncClient для общения с бэкендом.

    Использование:
        async with BackendAPIClient(base_url, timeout=30) as client:
            response = await client.chat(...)
    """

    def __init__(self, base_url: str, *, timeout_seconds: float = 60.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = httpx.Timeout(timeout_seconds, connect=5.0)
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> BackendAPIClient:
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _require_client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError(
                "BackendAPIClient must be used as 'async with' or via aclose-managed lifecycle"
            )
        return self._client

    async def chat(self, message: str, session_id: str | None) -> ChatResponse:
        """Обычный запрос — возвращает готовый ChatResponse."""
        client = self._require_client()
        payload = ChatRequest(message=message, session_id=session_id).model_dump(
            exclude_none=True
        )

        try:
            response = await client.post("/chat", json=payload)
        except httpx.TimeoutException as exc:
            raise BackendAPIError("Бэкенд не ответил вовремя.") from exc
        except httpx.HTTPError as exc:
            raise BackendAPIError(f"Сетевая ошибка при обращении к бэкенду: {exc}") from exc

        if response.status_code >= 400:
            raise BackendAPIError(
                f"Бэкенд вернул ошибку {response.status_code}: {response.text[:200]}",
                status_code=response.status_code,
            )

        try:
            return ChatResponse.model_validate(response.json())
        except (ValueError, json.JSONDecodeError) as exc:
            raise BackendAPIError(f"Невалидный ответ бэкенда: {exc}") from exc

    async def stream_chat(
        self, message: str, session_id: str | None
    ) -> AsyncIterator[StreamEvent]:
        """
        Стриминговый запрос — асинхронно отдаёт StreamEvent по мере прихода.

        Yields в порядке: session → token* → sources → done, либо error.
        """
        client = self._require_client()
        payload = ChatRequest(message=message, session_id=session_id).model_dump(
            exclude_none=True
        )

        try:
            async with client.stream("POST", "/chat/stream", json=payload) as response:
                if response.status_code >= 400:
                    body = await response.aread()
                    snippet = body.decode(errors="replace")[:200]
                    raise BackendAPIError(
                        f"Бэкенд вернул ошибку {response.status_code}: {snippet}",
                        status_code=response.status_code,
                    )

                async for event in _parse_sse_stream(response.aiter_lines()):
                    yield event
        except httpx.TimeoutException as exc:
            raise BackendAPIError("Бэкенд не ответил вовремя.") from exc
        except httpx.HTTPError as exc:
            raise BackendAPIError(f"Сетевая ошибка при обращении к бэкенду: {exc}") from exc


async def _parse_sse_stream(lines: AsyncIterator[str]) -> AsyncIterator[StreamEvent]:
    """
    Разбирает SSE-поток. Каждое событие — последовательность строк
    `event: <name>` и `data: <json>`, разделённых пустой строкой.

    Формат, который генерирует бэкенд, см. _sse_event в app/services/chat_service.py.
    """
    event_name: str | None = None
    data_buffer: list[str] = []

    async for raw_line in lines:
        line = raw_line.rstrip("\r")

        if line == "":
            if event_name and data_buffer:
                yield _build_event(event_name, "\n".join(data_buffer))
            event_name = None
            data_buffer = []
            continue

        if line.startswith(":"):
            # SSE comment — игнорируем
            continue

        if line.startswith("event:"):
            event_name = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_buffer.append(line[len("data:"):].lstrip(" "))

    # Финальный буфер без замыкающей пустой строки
    if event_name and data_buffer:
        yield _build_event(event_name, "\n".join(data_buffer))


def _build_event(event_name: str, raw_data: str) -> StreamEvent:
    try:
        data = json.loads(raw_data) if raw_data else {}
    except json.JSONDecodeError:
        logger.warning("Failed to parse SSE data for event %s: %r", event_name, raw_data[:200])
        data = {"raw": raw_data}
    if not isinstance(data, dict):
        data = {"value": data}
    return StreamEvent(type=event_name, data=data)


def parse_sources(raw: list[dict[str, Any]] | None) -> list[Source]:
    """Восстанавливает Source-модели из payload событий sources/done."""
    if not raw:
        return []
    parsed: list[Source] = []
    for item in raw:
        try:
            parsed.append(Source.model_validate(item))
        except ValueError:
            logger.warning("Skipping malformed source payload: %r", item)
    return parsed
