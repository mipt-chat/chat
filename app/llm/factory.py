"""
Фабрика LLM-провайдера.
Создаёт и кэширует singleton-экземпляр провайдера на время жизни приложения.
"""

from app.core import get_logger
from app.core.config import settings
from app.llm.auth import GigaChatOAuthAuth, LLMAuthProvider, StaticApiKeyAuth
from app.llm.base import BaseLLMProvider
from app.llm.provider import OpenAICompatibleProvider

logger = get_logger(__name__)

_provider: BaseLLMProvider | None = None


def _plain_api_key(raw_key: object) -> str:
    if hasattr(raw_key, "get_secret_value"):
        return raw_key.get_secret_value() or "no-key"
    return str(raw_key) if raw_key else "no-key"


def _model_name_for_provider(provider_name: str, config: dict) -> str:
    model_name = str(config["model_name"])

    if provider_name == "yandex":
        folder_id = config.get("folder_id")
        if folder_id:
            model_name = f"gpt://{folder_id}/{model_name}/latest"
            logger.debug("YandexGPT model URI assembled: %s", model_name)
    elif provider_name == "giga" and model_name.lower() == "gigachat":
        model_name = "GigaChat"

    return model_name


def _auth_for_provider(provider_name: str, config: dict, verify_ssl: bool) -> LLMAuthProvider:
    api_key = _plain_api_key(config.get("api_key"))

    if provider_name == "giga" and bool(config.get("use_oauth", True)):
        return GigaChatOAuthAuth(
            credentials=api_key,
            auth_url=str(config.get("auth_url") or ""),
            scope=str(config.get("scope") or "GIGACHAT_API_PERS"),
            verify_ssl=verify_ssl,
        )

    return StaticApiKeyAuth(api_key)


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
        provider_name = settings.active_llm_provider
        config = settings.get_active_provider_config()
        verify_ssl = bool(config.get("verify_ssl", True))
        _provider = OpenAICompatibleProvider(
            provider_name=provider_name,
            base_url=str(config["base_url"]),
            model_name=_model_name_for_provider(provider_name, config),
            auth=_auth_for_provider(provider_name, config, verify_ssl),
            verify_ssl=verify_ssl,
            fallback_answer=settings.fallback_answer,
            max_history_length=settings.max_history_length,
        )

    return _provider
