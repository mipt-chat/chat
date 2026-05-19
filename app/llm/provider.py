"""
Единственный конкретный LLM-провайдер проекта.
Использует OpenAI SDK с настраиваемым base_url, что позволяет работать
с YandexGPT, GigaChat и любым другим OpenAI-совместимым API.
"""

from openai import AsyncOpenAI

from app.core import get_logger
from app.core.config import settings
from app.core.exceptions import LLMProviderError
from app.llm.base import BaseLLMProvider
from app.llm.prompt import SCORE_THRESHOLD, build_prompt
from app.models.knowledge import RetrievedChunk
from app.models.session import DialogMessage

logger = get_logger(__name__)


class OpenAICompatibleProvider(BaseLLMProvider):
    """
    LLM-провайдер на основе OpenAI SDK.

    YandexGPT и GigaChat поддерживают OpenAI-совместимый формат —
    отличие только в base_url, api_key и model_name, которые берутся
    из settings.get_active_provider_config().

    Примечание по GigaChat: GigaChat использует временные OAuth-токены (TTL ~30 мин).
    В MVP передаётся статический токен через settings.gigachat_api_key.
    В production необходим refresh-механизм через /api/v2/oauth.

    Примечание по YandexGPT: имя модели собирается как gpt://{folder_id}/{model}/latest,
    если в конфиге задан folder_id. Base_url должен указывать на OpenAI-совместимый
    эндпоинт (.../foundationModels/v1/), а не на нативный (.../completions).
    """

    def __init__(self) -> None:
        config = settings.get_active_provider_config()

        # api_key может быть SecretStr (pydantic-settings) или plain str —
        # обрабатываем оба варианта, чтобы AsyncOpenAI получил реальный токен,
        # а не строку вида "**********".
        raw_key = config.get("api_key")
        if hasattr(raw_key, "get_secret_value"):
            api_key: str = raw_key.get_secret_value() or "no-key"
        else:
            api_key = str(raw_key) if raw_key else "no-key"

        base_url: str = config["base_url"]
        model_name: str = config["model_name"]

        # Для YandexGPT OpenAI-совместимый API требует model URI вида:
        # gpt://{folder_id}/{model_name}/latest
        if settings.active_llm_provider == "yandex":
            folder_id: str | None = config.get("folder_id")
            if folder_id:
                model_name = f"gpt://{folder_id}/{model_name}/latest"
                logger.debug("YandexGPT model URI assembled: %s", model_name)

        self._model_name = model_name
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

        logger.info(
            "LLM provider initialized: provider=%s model=%s base_url=%s",
            settings.active_llm_provider,
            self._model_name,
            base_url,
        )

    async def generate(
        self,
        question: str,
        context_chunks: list[RetrievedChunk],
        history: list[DialogMessage],
    ) -> tuple[str, bool]:
        """
        Генерирует ответ на вопрос пользователя.

        Если ни один чанк не прошёл фильтр по SCORE_THRESHOLD —
        возвращает (settings.fallback_answer, False) без обращения к API.

        История обрезается до settings.max_history_length последних сообщений.
        """
        system_prompt, answered = build_prompt(question, context_chunks)

        if not answered:
            logger.info("No relevant chunks above threshold, returning fallback answer")
            return settings.fallback_answer, False

        # Берём последние N сообщений — самый свежий контекст важнее начала диалога
        max_hist = settings.max_history_length
        trimmed_history = history[-max_hist:] if max_hist > 0 else []

        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        messages.extend({"role": msg.role, "content": msg.content} for msg in trimmed_history)
        messages.append({"role": "user", "content": question})

        logger.debug(
            "Sending request to LLM: model=%s history_len=%d relevant_chunks=%d total_chunks=%d",
            self._model_name,
            len(trimmed_history),
            len([c for c in context_chunks if c.score >= SCORE_THRESHOLD]),
            len(context_chunks),
        )

        try:
            response = await self._client.chat.completions.create(
                model=self._model_name,
                messages=messages,  # type: ignore[arg-type]
            )
            answer = response.choices[0].message.content or settings.fallback_answer
            logger.info("LLM response received, length=%d chars", len(answer))
            return answer, True

        except Exception as exc:
            logger.error("LLM request failed: %s", exc)
            raise LLMProviderError(
                message="Ошибка при обращении к LLM-провайдеру",
                detail=str(exc),
            ) from exc
