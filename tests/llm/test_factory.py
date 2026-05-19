"""
Тесты для app/llm/factory.py.
Проверяют singleton-поведение и корректность возвращаемого типа.
"""

from unittest.mock import MagicMock, patch

import pytest

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

        with patch("app.llm.factory.OpenAICompatibleProvider", return_value=MagicMock(spec=BaseLLMProvider)):
            second = get_llm_provider()

        assert first is not second
