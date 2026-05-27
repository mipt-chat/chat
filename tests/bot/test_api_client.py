"""Unit-тесты для app/bot/api_client.py."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx
import pytest

from app.bot.api_client import BackendAPIClient, BackendAPIError, StreamEvent, parse_sources


def _make_client(handler) -> BackendAPIClient:
    """Создаёт BackendAPIClient с подменённым httpx-транспортом."""
    transport = httpx.MockTransport(handler)
    client = BackendAPIClient(base_url="http://api.test", timeout_seconds=5.0)
    client._client = httpx.AsyncClient(  # type: ignore[attr-defined]
        base_url="http://api.test",
        transport=transport,
        timeout=httpx.Timeout(5.0, connect=2.0),
    )
    return client


def _sse_payload(*events: tuple[str, dict]) -> bytes:
    chunks: list[str] = []
    for name, data in events:
        chunks.append(f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n")
    return "".join(chunks).encode("utf-8")


@pytest.mark.asyncio
async def test_chat_returns_validated_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/chat"
        body = json.loads(request.content.decode())
        assert body == {"message": "привет", "session_id": "tg_user_1"}
        return httpx.Response(
            200,
            json={
                "answer": "Здравствуйте!",
                "sources": [],
                "answered": True,
                "session_id": "tg_user_1",
            },
        )

    client = _make_client(handler)
    try:
        response = await client.chat(message="привет", session_id="tg_user_1")
    finally:
        await client.aclose()

    assert response.answer == "Здравствуйте!"
    assert response.session_id == "tg_user_1"
    assert response.answered is True


@pytest.mark.asyncio
async def test_chat_raises_backend_api_error_on_5xx() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="LLM down")

    client = _make_client(handler)
    try:
        with pytest.raises(BackendAPIError) as info:
            await client.chat(message="x", session_id=None)
    finally:
        await client.aclose()

    assert info.value.status_code == 503


@pytest.mark.asyncio
async def test_chat_omits_none_session_id_from_payload() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(
            200,
            json={"answer": "ok", "sources": [], "answered": True, "session_id": "web_x"},
        )

    client = _make_client(handler)
    try:
        await client.chat(message="hi", session_id=None)
    finally:
        await client.aclose()

    assert captured["body"] == {"message": "hi"}


@pytest.mark.asyncio
async def test_chat_raises_on_invalid_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json")

    client = _make_client(handler)
    try:
        with pytest.raises(BackendAPIError):
            await client.chat(message="x", session_id=None)
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_chat_raises_on_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out", request=request)

    client = _make_client(handler)
    try:
        with pytest.raises(BackendAPIError):
            await client.chat(message="x", session_id=None)
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_stream_chat_yields_events_in_order() -> None:
    payload = _sse_payload(
        ("session", {"session_id": "tg_user_42_stream"}),
        ("token", {"text": "Привет"}),
        ("token", {"text": ", мир"}),
        ("sources", {"sources": []}),
        (
            "done",
            {
                "answer": "Привет, мир",
                "answered": True,
                "session_id": "tg_user_42_stream",
                "sources": [],
            },
        ),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/chat/stream"
        return httpx.Response(
            200,
            content=payload,
            headers={"content-type": "text/event-stream"},
        )

    client = _make_client(handler)
    events: list[StreamEvent] = []
    try:
        async for event in client.stream_chat(message="привет", session_id="tg_user_42_stream"):
            events.append(event)
    finally:
        await client.aclose()

    assert [e.type for e in events] == ["session", "token", "token", "sources", "done"]
    assert events[1].data["text"] == "Привет"
    assert events[-1].data["answer"] == "Привет, мир"


@pytest.mark.asyncio
async def test_stream_chat_raises_on_4xx() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, text="bad request")

    client = _make_client(handler)

    async def consume() -> AsyncIterator[StreamEvent]:
        async for event in client.stream_chat(message="hello", session_id=None):
            yield event

    try:
        with pytest.raises(BackendAPIError) as info:
            async for _ in consume():
                pass
    finally:
        await client.aclose()

    assert info.value.status_code == 422


def test_parse_sources_skips_invalid_items() -> None:
    raw = [
        {
            "chunk_id": "c1",
            "source_path": "instructions/a.txt",
            "text": "...",
            "score": 0.9,
        },
        {"not": "a source"},
    ]
    parsed = parse_sources(raw)
    assert len(parsed) == 1
    assert parsed[0].source_path == "instructions/a.txt"


def test_parse_sources_handles_none() -> None:
    assert parse_sources(None) == []
    assert parse_sources([]) == []


def test_require_client_raises_without_context() -> None:
    client = BackendAPIClient(base_url="http://api.test")
    with pytest.raises(RuntimeError):
        client._require_client()  # noqa: SLF001 — целенаправленно тестируем guard
