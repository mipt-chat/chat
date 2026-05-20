"""
Тесты для app/llm/factory.py.
Проверяют singleton-поведение и корректность возвращаемого типа.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.llm.auth import GigaChatOAuthAuth, StaticApiKeyAuth
from app.llm.base import BaseLLMProvider


@pytest.fixture(autouse=True)
def reset_singleton():
    """Сбрасывает singleton перед каждым тестом для изоляции."""
    import app.llm.factory as factory_module
    factory_module._provider = None
    yield
    factory_module._provider = None


@pytest.fixture()
def mock_provider():
    """Подменяет OpenAICompatibleProvider на mock, чтобы не инициализировать реальный клиент."""
    provider_mock = MagicMock(spec=BaseLLMProvider)
    with patch("app.llm.factory.OpenAICompatibleProvider", return_value=provider_mock):
        yield provider_mock


class TestGetLlmProvider:
    def test_returns_base_provider_instance(self, mock_provider):
        from app.llm.factory import get_llm_provider
        result = get_llm_provider()
        assert isinstance(result, BaseLLMProvider)

    def test_returns_same_instance_on_repeated_calls(self, mock_provider):
        from app.llm.factory import get_llm_provider
        first = get_llm_provider()
        second = get_llm_provider()
        assert first is second

    def test_provider_created_only_once(self, mock_provider):
        from app.llm import get_llm_provider
        with patch("app.llm.factory.OpenAICompatibleProvider", return_value=mock_provider) as cls_mock:
            get_llm_provider()
            get_llm_provider()
            get_llm_provider()
            cls_mock.assert_called_once()

    def test_reexport_from_package_init(self, mock_provider):
        """get_llm_provider должен быть доступен через app.llm напрямую."""
        from app.llm import get_llm_provider
        result = get_llm_provider()
        assert result is not None

    def test_singleton_reset_creates_new_instance(self, mock_provider):
        import app.llm.factory as factory_module
        from app.llm.factory import get_llm_provider

        first = get_llm_provider()
        factory_module._provider = None  # эмулируем сброс (например, при тестах)

        with patch(
            "app.llm.factory.OpenAICompatibleProvider",
            return_value=MagicMock(spec=BaseLLMProvider),
        ):
            second = get_llm_provider()

        assert first is not second

    def test_yandex_model_uri_assembled_with_folder_id(self):
        from app.llm.factory import _model_name_for_provider

        model_name = _model_name_for_provider(
            "yandex",
            {
                "model_name": "yandexgpt-lite",
                "folder_id": "b1g123folder",
            },
        )

        assert model_name == "gpt://b1g123folder/yandexgpt-lite/latest"

    def test_yandex_model_name_unchanged_without_folder_id(self):
        from app.llm.factory import _model_name_for_provider

        model_name = _model_name_for_provider(
            "yandex",
            {"model_name": "yandexgpt-lite"},
        )

        assert model_name == "yandexgpt-lite"

    def test_gigachat_model_name_normalized(self):
        from app.llm.factory import _model_name_for_provider

        assert _model_name_for_provider("giga", {"model_name": "gigachat"}) == "GigaChat"

    def test_secret_str_api_key_is_unwrapped(self):
        from app.llm.factory import _plain_api_key

        secret = MagicMock()
        secret.get_secret_value.return_value = "real-secret-token"

        assert _plain_api_key(secret) == "real-secret-token"
        secret.get_secret_value.assert_called_once()

    def test_plain_str_api_key_works_without_unwrap(self):
        from app.llm.factory import _plain_api_key

        assert _plain_api_key("plain-token") == "plain-token"

    def test_openai_compatible_uses_static_auth(self, monkeypatch):
        settings_mock = MagicMock()
        settings_mock.active_llm_provider = "openai_compatible"
        settings_mock.fallback_answer = "fallback"
        settings_mock.max_history_length = 5
        settings_mock.llm_request_timeout_seconds = 600.0
        settings_mock.llm_connect_timeout_seconds = 5.0
        settings_mock.get_active_provider_config.return_value = {
            "api_key": "plain-token",
            "base_url": "https://example.com/v1",
            "model_name": "model",
        }

        monkeypatch.setattr("app.llm.factory.settings", settings_mock)

        from app.llm.factory import get_llm_provider

        with patch("app.llm.factory.OpenAICompatibleProvider") as provider_cls:
            get_llm_provider()

        kwargs = provider_cls.call_args.kwargs
        assert kwargs["provider_name"] == "openai_compatible"
        assert kwargs["model_name"] == "model"
        assert isinstance(kwargs["auth"], StaticApiKeyAuth)
        assert kwargs["request_timeout_seconds"] == 600.0
        assert kwargs["connect_timeout_seconds"] == 5.0

    def test_gigachat_uses_oauth_auth_when_enabled(self, monkeypatch):
        settings_mock = MagicMock()
        settings_mock.active_llm_provider = "giga"
        settings_mock.fallback_answer = "fallback"
        settings_mock.max_history_length = 5
        settings_mock.llm_request_timeout_seconds = 600.0
        settings_mock.llm_connect_timeout_seconds = 5.0
        settings_mock.get_active_provider_config.return_value = {
            "api_key": "credentials",
            "base_url": "https://gigachat.devices.sberbank.ru/api/v1",
            "model_name": "gigachat",
            "auth_url": "https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
            "scope": "GIGACHAT_API_PERS",
            "use_oauth": True,
            "verify_ssl": False,
        }

        monkeypatch.setattr("app.llm.factory.settings", settings_mock)

        from app.llm.factory import get_llm_provider

        with patch("app.llm.factory.OpenAICompatibleProvider") as provider_cls:
            get_llm_provider()

        kwargs = provider_cls.call_args.kwargs
        assert kwargs["provider_name"] == "giga"
        assert kwargs["model_name"] == "GigaChat"
        assert isinstance(kwargs["auth"], GigaChatOAuthAuth)

    @pytest.mark.asyncio
    async def test_close_llm_provider_closes_and_resets_singleton(self):
        import app.llm.factory as factory_module
        from app.llm.factory import close_llm_provider

        provider = MagicMock(spec=BaseLLMProvider)
        provider.close = AsyncMock()
        factory_module._provider = provider

        await close_llm_provider()

        provider.close.assert_awaited_once()
        assert factory_module._provider is None
