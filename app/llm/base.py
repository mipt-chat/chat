"""
Абстрактный базовый класс для LLM-провайдеров.
Все конкретные провайдеры должны наследоваться от BaseLLMProvider.
"""

from abc import ABC, abstractmethod

from app.models.knowledge import RetrievedChunk
from app.models.session import DialogMessage


class BaseLLMProvider(ABC):
    """
    Абстракция над LLM-провайдером.
    Позволяет переключаться между YandexGPT, GigaChat и другими
    провайдерами без изменений в остальной системе.
    """

    @abstractmethod
    async def generate(
        self,
        question: str,
        context_chunks: list[RetrievedChunk],
        history: list[DialogMessage],
    ) -> tuple[str, bool]:
        """
        Генерирует ответ на вопрос пользователя с учётом контекста и истории.

        Args:
            question: Вопрос пользователя.
            context_chunks: Список релевантных чанков из RAG-слоя.
            history: История диалога текущей сессии.

        Returns:
            Кортеж (answer, answered), где:
              - answer: текст ответа (или fallback, если ответ не найден);
              - answered: True если найден релевантный контекст, False иначе.

        Raises:
            LLMProviderError: при ошибке обращения к провайдеру.
        """
        ...
