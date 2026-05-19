"""
Фабрика LLM-провайдера.
Создаёт и кэширует singleton-экземпляр провайдера на время жизни приложения.
"""

from app.core import get_logger
from app.llm.base import BaseLLMProvider
from app.llm.provider import OpenAICompatibleProvider

logger = get_logger(__name__)

_provider: BaseLLMProvider | None = None


def get_llm_provider() -> BaseLLMProvider:
    """
    Возвращает singleton-экземпляр активного LLM-провайдера.

    Провайдер создаётся при первом вызове и переиспользуется далее.
    Конкретный класс провайдера определяется через settings.active_llm_provider.

    Рекомендуемый способ инициализации — в lifespan FastAPI-приложения:

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            app.state.llm = get_llm_provider()
            yield

    Returns:
        Экземпляр BaseLLMProvider, готовый к вызову generate().

    Raises:
        LLMProviderError: если конфигурация провайдера некорректна.
    """
    global _provider

    if _provider is None:
        logger.info("Creating LLM provider singleton")
        _provider = OpenAICompatibleProvider()

    return _provider
