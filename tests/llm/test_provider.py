"""
Тесты для app/llm/provider.py.
Используют моки AsyncOpenAI, чтобы не обращаться к реальным API.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.exceptions import LLMProviderError
from app.models.knowledge import RetrievedChunk
from app.models.session import DialogMessage


def _make_chunk(text: str, score: float, chunk_id: str = "c1") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        metadata={"source_path": "docs/test.txt"},
        score=score,
    )


def _make_message(role: str, content: str) -> DialogMessage:
    from datetime import datetime
    return DialogMessage(role=role, content=content, timestamp=datetime.now())


def _make_completion_response(text: str) -> MagicMock:
    """Собирает mock-ответ в форме openai.types.chat.ChatCompletion."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = text
    return response


# ---------------------------------------------------------------------------
# Вспомогательный fixture: мокаем AsyncOpenAI и settings при каждом тесте
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_settings():
    settings_mock = MagicMock()
    settings_mock.active_llm_provider = "openai_compatible"
    settings_mock.get_active_provider_config.return_value = {
        "api_key": "test-key",
        "base_url": "https://example.com/v1",
        "model_name": "test-model",
    }
    settings_mock.fallback_answer = "Извините, я не нашёл ответа."
    settings_mock.max_history_length = 10
    return settings_mock


@pytest.fixture()
def mock_openai_client():
    with patch("app.llm.provider.AsyncOpenAI") as mock_cls:
        client = MagicMock()
        mock_cls.return_value = client
        yield client


def _provider(settings_mock):
    from app.llm.auth import StaticApiKeyAuth
    from app.llm.provider import OpenAICompatibleProvider

    config = settings_mock.get_active_provider_config.return_value
    return OpenAICompatibleProvider(
        provider_name=settings_mock.active_llm_provider,
        base_url=config.get("base_url", "https://example.com/v1"),
        model_name=config.get("model_name", "test-model"),
        auth=StaticApiKeyAuth(str(config.get("api_key") or "no-key")),
        verify_ssl=bool(config.get("verify_ssl", True)),
        fallback_answer=settings_mock.fallback_answer,
        max_history_length=settings_mock.max_history_length,
    )


# ---------------------------------------------------------------------------
# Тесты инициализации провайдера
# ---------------------------------------------------------------------------

class TestProviderInit:
    def test_openai_compatible_init(self, mock_settings, mock_openai_client):
        provider = _provider(mock_settings)
        assert provider._model_name == "test-model"

    def test_model_name_is_used_as_passed(self, mock_settings, mock_openai_client):
        mock_settings.active_llm_provider = "yandex"
        mock_settings.get_active_provider_config.return_value = {
            "api_key": "yc-key",
            "base_url": "https://llm.api.cloud.yandex.net/foundationModels/v1/",
            "model_name": "yandexgpt-lite",
            "folder_id": "b1g123folder",
        }
        provider = _provider(mock_settings)
        assert provider._model_name == "yandexgpt-lite"

    def test_missing_api_key_defaults_to_no_key(self, mock_settings, mock_openai_client):
        mock_settings.get_active_provider_config.return_value = {
            "base_url": "https://example.com/v1",
            "model_name": "model",
        }
        provider = _provider(mock_settings)
        # Не должно выбрасывать исключение
        assert provider._model_name == "model"


# ---------------------------------------------------------------------------
# Тесты метода generate()
# ---------------------------------------------------------------------------

class TestProviderGenerate:
    @pytest.mark.asyncio
    async def test_returns_fallback_when_no_relevant_chunks(self, mock_settings, mock_openai_client):
        provider = _provider(mock_settings)
        low_score_chunks = [_make_chunk("текст", score=0.1)]
        answer, answered = await provider.generate("вопрос", low_score_chunks, [])
        assert answered is False
        assert answer == mock_settings.fallback_answer

    @pytest.mark.asyncio
    async def test_no_api_call_when_no_relevant_chunks(self, mock_settings, mock_openai_client):
        provider = _provider(mock_settings)
        low_score_chunks = [_make_chunk("текст", score=0.0)]
        await provider.generate("вопрос", low_score_chunks, [])
        mock_openai_client.chat.completions.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_llm_answer_when_relevant_chunks(self, mock_settings, mock_openai_client):
        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=_make_completion_response("Ответ от LLM")
        )
        provider = _provider(mock_settings)
        chunks = [_make_chunk("релевантный контекст", score=0.8)]
        answer, answered = await provider.generate("вопрос", chunks, [])
        assert answered is True
        assert answer == "Ответ от LLM"

    @pytest.mark.asyncio
    async def test_history_is_trimmed_to_max_length(self, mock_settings, mock_openai_client):
        mock_settings.max_history_length = 2
        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=_make_completion_response("ok")
        )
        provider = _provider(mock_settings)
        history = [_make_message("user", f"сообщение {i}") for i in range(5)]
        chunks = [_make_chunk("контекст", score=0.9)]
        await provider.generate("вопрос", chunks, history)

        call_args = mock_openai_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        # system + 2 history + 1 user = 4
        assert len(messages) == 4

    @pytest.mark.asyncio
    async def test_empty_history_works(self, mock_settings, mock_openai_client):
        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=_make_completion_response("ответ")
        )
        provider = _provider(mock_settings)

        answer, answered = await provider.generate("вопрос", [], [])
        # Нет релевантных чанков → fallback
        assert answered is False

    @pytest.mark.asyncio
    async def test_messages_order_system_history_user(self, mock_settings, mock_openai_client):
        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=_make_completion_response("ок")
        )
        provider = _provider(mock_settings)
        history = [_make_message("assistant", "предыдущий ответ")]
        chunks = [_make_chunk("контекст", score=0.9)]
        await provider.generate("новый вопрос", chunks, history)

        messages = mock_openai_client.chat.completions.create.call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "assistant"
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "новый вопрос"

    @pytest.mark.asyncio
    async def test_api_error_raises_llm_provider_error(self, mock_settings, mock_openai_client):
        mock_openai_client.chat.completions.create = AsyncMock(
            side_effect=Exception("connection timeout")
        )
        provider = _provider(mock_settings)
        chunks = [_make_chunk("контекст", score=0.9)]
        with pytest.raises(LLMProviderError):
            await provider.generate("вопрос", chunks, [])

    @pytest.mark.asyncio
    async def test_none_content_falls_back_to_fallback_answer(self, mock_settings, mock_openai_client):
        """Если LLM вернул None в content — используем fallback и считаем ответ ненайденным."""
        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=_make_completion_response(None)
        )
        provider = _provider(mock_settings)
        chunks = [_make_chunk("контекст", score=0.9)]
        answer, answered = await provider.generate("вопрос", chunks, [])
        assert answered is False
        assert answer == mock_settings.fallback_answer

    @pytest.mark.asyncio
    async def test_no_answer_phrase_marks_unanswered(self, mock_settings, mock_openai_client):
        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=_make_completion_response("База знаний не содержит информации об этом.")
        )
        provider = _provider(mock_settings)
        chunks = [_make_chunk("контекст", score=0.9)]
        answer, answered = await provider.generate("вопрос", chunks, [])
        assert answered is False
        assert answer == "База знаний не содержит информации об этом."


