"""
Единственный конкретный LLM-провайдер проекта.
Использует OpenAI SDK с настраиваемым base_url и auth-стратегией,
что позволяет работать с YandexGPT, GigaChat и любым другим
OpenAI-совместимым API.
"""

from asyncio import Lock
from collections.abc import AsyncIterator
from inspect import isawaitable

import httpx
from openai import AsyncOpenAI

from app.core import get_logger
from app.core.exceptions import LLMProviderError
from app.llm.auth import LLMAuthProvider
from app.llm.base import BaseLLMProvider, LLMStreamChunk
from app.llm.prompt import SCORE_THRESHOLD, build_prompt
from app.models.knowledge import RetrievedChunk
from app.models.session import DialogMessage

logger = get_logger(__name__)


def _is_no_answer(answer: str) -> bool:
    normalized = answer.lower()
    no_answer_markers = (
        "база знаний не содержит",
        "контекст не содержит",
        "нет информации",
        "не нашёл ответ",
        "не нашла ответ",
        "не содержит ответа",
        "не могу ответить на основе",
    )
    return any(marker in normalized for marker in no_answer_markers)


class OpenAICompatibleProvider(BaseLLMProvider):
    """
    LLM-провайдер на основе OpenAI SDK.

    Класс намеренно не знает деталей конкретного провайдера: OAuth, model URI,
    нормализация имён моделей и прочие особенности собираются в factory/config.
    """

    def __init__(
        self,
        *,
        provider_name: str,
        base_url: str,
        model_name: str,
        auth: LLMAuthProvider,
        verify_ssl: bool = True,
        fallback_answer: str,
        max_history_length: int,
        request_timeout_seconds: float,
        connect_timeout_seconds: float,
    ) -> None:
        self._provider_name = provider_name
        self._base_url = base_url
        self._verify_ssl = verify_ssl
        self._auth = auth
        self._fallback_answer = fallback_answer
        self._max_history_length = max_history_length
        self._timeout = httpx.Timeout(
            timeout=request_timeout_seconds,
            connect=connect_timeout_seconds,
        )
        self._model_name = model_name
        self._client: AsyncOpenAI | None = None
        self._client_api_key: str | None = None
        self._client_lock = Lock()

        logger.info(
            "LLM provider initialized: provider=%s model=%s base_url=%s verify_ssl=%s",
            provider_name,
            self._model_name,
            base_url,
            verify_ssl,
        )

    def _make_client(self, api_key: str) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key=api_key,
            base_url=self._base_url,
            timeout=self._timeout,
            http_client=httpx.AsyncClient(
                verify=self._verify_ssl,
                timeout=self._timeout,
            ),
        )

    async def _get_client(self) -> AsyncOpenAI:
        api_key = await self._auth.get_api_key()
        async with self._client_lock:
            if self._client is None or self._client_api_key != api_key:
                old_client = self._client
                self._client = self._make_client(api_key)
                self._client_api_key = api_key
                await self._close_client(old_client)
            return self._client

    async def close(self) -> None:
        async with self._client_lock:
            client = self._client
            self._client = None
            self._client_api_key = None
            await self._close_client(client)

    @staticmethod
    async def _close_client(client: AsyncOpenAI | None) -> None:
        if client is None:
            return
        result = client.close()
        if isawaitable(result):
            await result

    def _build_messages(
        self,
        question: str,
        context_chunks: list[RetrievedChunk],
        history: list[DialogMessage],
    ) -> tuple[list[dict[str, str]], bool, list[DialogMessage]]:
        system_prompt, answered = build_prompt(question, context_chunks)

        if not answered:
            return [], False, []

        max_hist = self._max_history_length
        trimmed_history = history[-max_hist:] if max_hist > 0 else []

        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        messages.extend({"role": msg.role, "content": msg.content} for msg in trimmed_history)
        messages.append({"role": "user", "content": question})
        return messages, True, trimmed_history

    async def generate(
        self,
        question: str,
        context_chunks: list[RetrievedChunk],
        history: list[DialogMessage],
    ) -> tuple[str, bool]:
        """
        Генерирует ответ на вопрос пользователя.

        Если ни один чанк не прошёл фильтр по SCORE_THRESHOLD —
        возвращает fallback-ответ без обращения к API.

        История обрезается до настроенного max_history_length последних сообщений.
        """
        messages, answered, trimmed_history = self._build_messages(question, context_chunks, history)

        if not answered:
            logger.info("No relevant chunks above threshold, returning fallback answer")
            return self._fallback_answer, False

        logger.debug(
            "Sending request to LLM: model=%s history_len=%d relevant_chunks=%d total_chunks=%d",
            self._model_name,
            len(trimmed_history),
            len([c for c in context_chunks if c.score >= SCORE_THRESHOLD]),
            len(context_chunks),
        )

        try:
            client = await self._get_client()
            response = await client.chat.completions.create(
                model=self._model_name,
                messages=messages,  # type: ignore[arg-type]
            )
            answer = response.choices[0].message.content or self._fallback_answer
            logger.info("LLM response received, length=%d chars", len(answer))
            if _is_no_answer(answer):
                logger.info("LLM reported that context has no answer")
                return answer, False
            return answer, True

        except Exception as exc:
            logger.error("LLM request failed: %s", exc)
            raise LLMProviderError(
                message="Ошибка при обращении к LLM-провайдеру",
                detail=str(exc),
            ) from exc

    async def stream(
        self,
        question: str,
        context_chunks: list[RetrievedChunk],
        history: list[DialogMessage],
    ) -> AsyncIterator[LLMStreamChunk]:
        """Stream an LLM answer token-by-token where provider supports it."""

        messages, answered, trimmed_history = self._build_messages(question, context_chunks, history)

        if not answered:
            logger.info("No relevant chunks above threshold, streaming fallback answer")
            yield LLMStreamChunk(text=self._fallback_answer)
            yield LLMStreamChunk(is_final=True, answered=False)
            return

        logger.debug(
            "Sending streaming request to LLM: model=%s history_len=%d "
            "relevant_chunks=%d total_chunks=%d",
            self._model_name,
            len(trimmed_history),
            len([c for c in context_chunks if c.score >= SCORE_THRESHOLD]),
            len(context_chunks),
        )

        answer_parts: list[str] = []
        try:
            client = await self._get_client()
            stream = await client.chat.completions.create(
                model=self._model_name,
                messages=messages,  # type: ignore[arg-type]
                stream=True,
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                token = chunk.choices[0].delta.content or ""
                if not token:
                    continue
                answer_parts.append(token)
                yield LLMStreamChunk(text=token)

            answer = "".join(answer_parts).strip()
            if not answer:
                answer = self._fallback_answer
                yield LLMStreamChunk(text=answer)

            logger.info("LLM streaming response received, length=%d chars", len(answer))
            yield LLMStreamChunk(is_final=True, answered=not _is_no_answer(answer))

        except Exception as exc:
            logger.error("LLM streaming request failed: %s", exc)
            raise LLMProviderError(
                message="Ошибка при обращении к LLM-провайдеру",
                detail=str(exc),
            ) from exc
