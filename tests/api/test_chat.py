from pathlib import Path

from fastapi.testclient import TestClient

from app.core.exceptions import LLMProviderError, RetrievalError
from app.llm.base import LLMStreamChunk
from app.main import create_app
from app.models.knowledge import RetrievedChunk
from app.models.session import DialogMessage
from app.services import chat_service


class _FakeSessionStore:
    def __init__(self) -> None:
        self.saved: list[tuple[str, str, str]] = []

    def get_recent_messages(self, session_id: str) -> list[DialogMessage]:
        return []

    def add_message(self, session_id: str, role: str, content: str) -> None:
        self.saved.append((session_id, role, content))


class _FakeMetricsCollector:
    def __init__(self) -> None:
        self.records: list[tuple[str, str | None, str]] = []

    def record_unanswered(self, question: str, session_id: str | None = None, reason: str = "") -> None:
        self.records.append((question, session_id, reason))


class _FakeProvider:
    def __init__(self, *, answered: bool = True, answer: str = "Ответ из LLM") -> None:
        self.answered = answered
        self.answer = answer
        self.calls: list[tuple[str, list[RetrievedChunk], list[DialogMessage]]] = []

    async def generate(
        self,
        question: str,
        context_chunks: list[RetrievedChunk],
        history: list[DialogMessage],
    ) -> tuple[str, bool]:
        self.calls.append((question, context_chunks, history))
        return self.answer, self.answered

    async def stream(
        self,
        question: str,
        context_chunks: list[RetrievedChunk],
        history: list[DialogMessage],
    ):
        self.calls.append((question, context_chunks, history))
        yield LLMStreamChunk(text=self.answer)
        yield LLMStreamChunk(is_final=True, answered=self.answered)


def _client() -> TestClient:
    return TestClient(create_app(), raise_server_exceptions=False)


def _chunk() -> RetrievedChunk:
    source_path = Path.cwd() / "knowledge_base" / "instructions" / "registration.fiz.txt"
    return RetrievedChunk(
        chunk_id=f"{source_path}:0",
        text="Для регистрации физического лица откройте форму регистрации.",
        metadata={"source_path": str(source_path), "chunk_index": 0},
        score=0.91,
    )


def test_chat_success(monkeypatch) -> None:
    fake_session_store = _FakeSessionStore()
    fake_metrics = _FakeMetricsCollector()
    fake_provider = _FakeProvider(answer="Готовый ответ", answered=True)

    monkeypatch.setattr(chat_service, "session_store", fake_session_store)
    monkeypatch.setattr(chat_service, "metrics_collector", fake_metrics)
    monkeypatch.setattr(chat_service, "get_llm_provider", lambda: fake_provider)
    monkeypatch.setattr(chat_service, "search_context", lambda question: [_chunk()])

    response = _client().post(
        "/chat",
        json={"message": "Как зарегистрироваться?", "session_id": "tg_user_1"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Готовый ответ"
    assert body["answered"] is True
    assert body["session_id"] == "tg_user_1"
    assert body["sources"][0]["source_path"] == "knowledge_base/instructions/registration.fiz.txt"
    assert fake_session_store.saved == [
        ("tg_user_1", "user", "Как зарегистрироваться?"),
        ("tg_user_1", "assistant", "Готовый ответ"),
    ]
    assert fake_metrics.records == []


def test_chat_generates_session_id_when_missing(monkeypatch) -> None:
    fake_session_store = _FakeSessionStore()
    monkeypatch.setattr(chat_service, "session_store", fake_session_store)
    monkeypatch.setattr(chat_service, "metrics_collector", _FakeMetricsCollector())
    monkeypatch.setattr(chat_service, "get_llm_provider", lambda: _FakeProvider())
    monkeypatch.setattr(chat_service, "search_context", lambda question: [])

    response = _client().post("/chat", json={"message": "Привет"})

    assert response.status_code == 200
    session_id = response.json()["session_id"]
    assert session_id.startswith("web_")
    assert fake_session_store.saved[0][0] == session_id


def test_chat_strips_message_before_processing(monkeypatch) -> None:
    fake_session_store = _FakeSessionStore()
    fake_provider = _FakeProvider()
    seen_questions: list[str] = []

    def search(question: str) -> list[RetrievedChunk]:
        seen_questions.append(question)
        return []

    monkeypatch.setattr(chat_service, "session_store", fake_session_store)
    monkeypatch.setattr(chat_service, "metrics_collector", _FakeMetricsCollector())
    monkeypatch.setattr(chat_service, "get_llm_provider", lambda: fake_provider)
    monkeypatch.setattr(chat_service, "search_context", search)

    response = _client().post("/chat", json={"message": "  Привет  ", "session_id": "web_trim"})

    assert response.status_code == 200
    assert seen_questions == ["Привет"]
    assert fake_session_store.saved[0] == ("web_trim", "user", "Привет")


def test_chat_rejects_whitespace_only_message() -> None:
    response = _client().post("/chat", json={"message": "   "})

    assert response.status_code == 422


def test_chat_stream_rejects_whitespace_only_message() -> None:
    response = _client().post("/chat/stream", json={"message": "\n\t "})

    assert response.status_code == 422


def test_chat_records_unanswered_metric(monkeypatch) -> None:
    fake_session_store = _FakeSessionStore()
    fake_metrics = _FakeMetricsCollector()
    monkeypatch.setattr(chat_service, "session_store", fake_session_store)
    monkeypatch.setattr(chat_service, "metrics_collector", fake_metrics)
    monkeypatch.setattr(
        chat_service,
        "get_llm_provider",
        lambda: _FakeProvider(answer="fallback", answered=False),
    )
    monkeypatch.setattr(chat_service, "search_context", lambda question: [])

    response = _client().post(
        "/chat",
        json={"message": "Вопрос без ответа", "session_id": "web_test"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answered"] is False
    assert body["answer"] == "fallback"
    assert fake_metrics.records == [("Вопрос без ответа", "web_test", "low_relevance")]


def test_chat_retrieval_error_returns_503(monkeypatch) -> None:
    def raise_retrieval_error(question: str) -> list[RetrievedChunk]:
        raise RetrievalError("Ошибка поиска", detail="chroma is down")

    monkeypatch.setattr(chat_service, "search_context", raise_retrieval_error)

    response = _client().post("/chat", json={"message": "test"})

    assert response.status_code == 503
    assert response.json() == {"error": "Ошибка поиска", "detail": "chroma is down"}


def test_chat_llm_error_returns_502(monkeypatch) -> None:
    class ErrorProvider:
        async def generate(self, *args, **kwargs):
            raise LLMProviderError("Ошибка LLM", detail="timeout")

    monkeypatch.setattr(chat_service, "session_store", _FakeSessionStore())
    monkeypatch.setattr(chat_service, "search_context", lambda question: [_chunk()])
    monkeypatch.setattr(chat_service, "get_llm_provider", lambda: ErrorProvider())

    response = _client().post("/chat", json={"message": "test"})

    assert response.status_code == 502
    assert response.json() == {"error": "Ошибка LLM", "detail": "timeout"}


def test_chat_stream_success(monkeypatch) -> None:
    fake_session_store = _FakeSessionStore()
    fake_provider = _FakeProvider(answer="Потоковый ответ", answered=True)

    monkeypatch.setattr(chat_service, "session_store", fake_session_store)
    monkeypatch.setattr(chat_service, "metrics_collector", _FakeMetricsCollector())
    monkeypatch.setattr(chat_service, "get_llm_provider", lambda: fake_provider)
    monkeypatch.setattr(chat_service, "search_context", lambda question: [_chunk()])

    response = _client().post(
        "/chat/stream",
        json={"message": "Как зарегистрироваться?", "session_id": "web_stream"},
    )

    assert response.status_code == 200
    body = response.text
    assert "event: session" in body
    assert 'data: {"text": "Потоковый ответ"}' in body
    assert "event: sources" in body
    assert '"answered": true' in body
    assert fake_session_store.saved == [
        ("web_stream", "user", "Как зарегистрироваться?"),
        ("web_stream", "assistant", "Потоковый ответ"),
    ]


def test_chat_stream_unanswered_hides_sources(monkeypatch) -> None:
    fake_metrics = _FakeMetricsCollector()

    monkeypatch.setattr(chat_service, "session_store", _FakeSessionStore())
    monkeypatch.setattr(chat_service, "metrics_collector", fake_metrics)
    monkeypatch.setattr(
        chat_service,
        "get_llm_provider",
        lambda: _FakeProvider(answer="Нет ответа", answered=False),
    )
    monkeypatch.setattr(chat_service, "search_context", lambda question: [_chunk()])

    response = _client().post(
        "/chat/stream",
        json={"message": "Нерелевантный вопрос", "session_id": "web_stream"},
    )

    assert response.status_code == 200
    assert '"answered": false' in response.text
    assert '"sources": []' in response.text
    assert fake_metrics.records == [("Нерелевантный вопрос", "web_stream", "no_answer")]
