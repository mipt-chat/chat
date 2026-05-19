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
def mock_settings(monkeypatch):
    settings_mock = MagicMock()
    settings_mock.active_llm_provider = "openai_compatible"
    settings_mock.get_active_provider_config.return_value = {
        "api_key": "test-key",
        "base_url": "https://example.com/v1",
        "model_name": "test-model",
    }
    settings_mock.fallback_answer = "Извините, я не нашёл ответа."
    settings_mock.max_history_length = 10
    monkeypatch.setattr("app.llm.provider.settings", settings_mock)
    return settings_mock


@pytest.fixture()
def mock_openai_client():
    with patch("app.llm.provider.AsyncOpenAI") as mock_cls:
        client = MagicMock()
        mock_cls.return_value = client
        yield client


# ---------------------------------------------------------------------------
# Тесты инициализации провайдера
# ---------------------------------------------------------------------------

class TestProviderInit:
    def test_openai_compatible_init(self, mock_settings, mock_openai_client):
        from app.llm.provider import OpenAICompatibleProvider
        provider = OpenAICompatibleProvider()
        assert provider._model_name == "test-model"

    def test_yandex_model_uri_assembled_with_folder_id(self, mock_settings, mock_openai_client):
        mock_settings.active_llm_provider = "yandex"
        mock_settings.get_active_provider_config.return_value = {
            "api_key": "yc-key",
            "base_url": "https://llm.api.cloud.yandex.net/foundationModels/v1/",
            "model_name": "yandexgpt-lite",
            "folder_id": "b1g123folder",
        }
        from app.llm.provider import OpenAICompatibleProvider
        provider = OpenAICompatibleProvider()
        assert provider._model_name == "gpt://b1g123folder/yandexgpt-lite/latest"

    def test_yandex_model_name_unchanged_without_folder_id(self, mock_settings, mock_openai_client):
        mock_settings.active_llm_provider = "yandex"
        mock_settings.get_active_provider_config.return_value = {
            "api_key": "yc-key",
            "base_url": "https://llm.api.cloud.yandex.net/foundationModels/v1/",
            "model_name": "yandexgpt-lite",
        }
        from app.llm.provider import OpenAICompatibleProvider
        provider = OpenAICompatibleProvider()
        assert provider._model_name == "yandexgpt-lite"

    def test_missing_api_key_defaults_to_no_key(self, mock_settings, mock_openai_client):
        mock_settings.get_active_provider_config.return_value = {
            "base_url": "https://example.com/v1",
            "model_name": "model",
        }
        from app.llm.provider import OpenAICompatibleProvider
        provider = OpenAICompatibleProvider()
        # Не должно выбрасывать исключение
        assert provider._model_name == "model"


# ---------------------------------------------------------------------------
# Тесты метода generate()
# ---------------------------------------------------------------------------

class TestProviderGenerate:
    @pytest.mark.asyncio
    async def test_returns_fallback_when_no_relevant_chunks(self, mock_settings, mock_openai_client):
        from app.llm.provider import OpenAICompatibleProvider
        provider = OpenAICompatibleProvider()
        low_score_chunks = [_make_chunk("текст", score=0.1)]
        answer, answered = await provider.generate("вопрос", low_score_chunks, [])
        assert answered is False
        assert answer == mock_settings.fallback_answer

    @pytest.mark.asyncio
    async def test_no_api_call_when_no_relevant_chunks(self, mock_settings, mock_openai_client):
        from app.llm.provider import OpenAICompatibleProvider
        provider = OpenAICompatibleProvider()
        low_score_chunks = [_make_chunk("текст", score=0.0)]
        await provider.generate("вопрос", low_score_chunks, [])
        mock_openai_client.chat.completions.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_llm_answer_when_relevant_chunks(self, mock_settings, mock_openai_client):
        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=_make_completion_response("Ответ от LLM")
        )
        from app.llm.provider import OpenAICompatibleProvider
        provider = OpenAICompatibleProvider()
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
        from app.llm.provider import OpenAICompatibleProvider
        provider = OpenAICompatibleProvider()
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
        from app.llm.provider import OpenAICompatibleProvider
        provider = OpenAICompatibleProvider()

        answer, answered = await provider.generate("вопрос", [], [])
        # Нет релевантных чанков → fallback
        assert answered is False

    @pytest.mark.asyncio
    async def test_messages_order_system_history_user(self, mock_settings, mock_openai_client):
        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=_make_completion_response("ок")
        )
        from app.llm.provider import OpenAICompatibleProvider
        provider = OpenAICompatibleProvider()
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
        from app.llm.provider import OpenAICompatibleProvider
        provider = OpenAICompatibleProvider()
        chunks = [_make_chunk("контекст", score=0.9)]
        with pytest.raises(LLMProviderError):
            await provider.generate("вопрос", chunks, [])

    @pytest.mark.asyncio
    async def test_none_content_falls_back_to_fallback_answer(self, mock_settings, mock_openai_client):
        """Если LLM вернул None в content — используем fallback, но answered=True."""
        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=_make_completion_response(None)
        )
        from app.llm.provider import OpenAICompatibleProvider
        provider = OpenAICompatibleProvider()
        chunks = [_make_chunk("контекст", score=0.9)]
        answer, answered = await provider.generate("вопрос", chunks, [])
        assert answered is True
        assert answer == mock_settings.fallback_answer


class TestProviderSecretStr:
    def test_secret_str_api_key_is_unwrapped(self, mock_settings, mock_openai_client):
        """SecretStr из pydantic-settings должен быть распакован через get_secret_value()."""
        secret = MagicMock()
        secret.get_secret_value.return_value = "real-secret-token"
        mock_settings.get_active_provider_config.return_value = {
            "api_key": secret,
            "base_url": "https://example.com/v1",
            "model_name": "model",
        }
        from app.llm.provider import OpenAICompatibleProvider
        OpenAICompatibleProvider()
        # get_secret_value() должен быть вызван ровно один раз при инициализации
        secret.get_secret_value.assert_called_once()

    def test_plain_str_api_key_works_without_unwrap(self, mock_settings, mock_openai_client):
        """Обычная строка (не SecretStr) должна работать без вызова get_secret_value."""
        mock_settings.get_active_provider_config.return_value = {
            "api_key": "plain-token",
            "base_url": "https://example.com/v1",
            "model_name": "model",
        }
        from app.llm.provider import OpenAICompatibleProvider
        # Не должно выбрасывать AttributeError
        provider = OpenAICompatibleProvider()
        assert provider._model_name == "model"
