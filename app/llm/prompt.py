"""
Построение системного промпта для LLM.
Включает фильтрацию чанков по порогу релевантности и формирование контекста.
"""

from app.models.knowledge import RetrievedChunk

# схожесть (similarity) в диапазоне 0.0..1.0, чем выше тем релевантнее.
# Чанки с score < SCORE_THRESHOLD отсекаются.
# TODO: согласовать финальное значение порога после тестов на реальных данных
# (текущий placeholder 0.4; типичный диапазон для e5-модели: 0.3–0.6).
SCORE_THRESHOLD: float = 0.4

_SYSTEM_TEMPLATE = (
    "Ты — ИИ-ассистент клиентской поддержки. "
    "Отвечай строго на основе предоставленного контекста из базы знаний. "
    "Если контекст не содержит ответа — честно сообщи об этом. "
    "Отвечай на русском языке, кратко и по существу.\n\n"
    "КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ:\n"
    "{context}"
)


def build_prompt(
    question: str,
    context_chunks: list[RetrievedChunk],
) -> tuple[str, bool]:
    """
    Формирует системный промпт с контекстом из релевантных чанков.

    Чанки фильтруются по SCORE_THRESHOLD. Если ни один чанк не прошёл
    фильтр — возвращается пустая строка и answered=False, что сигнализирует
    провайдеру вернуть fallback-ответ.

    Args:
        question: Вопрос пользователя (зарезервирован для будущих use-cases,
                  например query-aware контекста; сейчас вопрос передаётся
                  отдельным user-сообщением).
        context_chunks: Чанки из RAG-слоя с оценками релевантности.

    Returns:
        Кортеж (system_prompt, answered):
          - system_prompt: готовый системный промпт с инжектированным контекстом;
          - answered: False если все чанки ниже порога, иначе True.
    """
    relevant = [chunk for chunk in context_chunks if chunk.score >= SCORE_THRESHOLD]

    if not relevant:
        return "", False

    context_parts = [
        f"[{i + 1}] {chunk.text}"
        for i, chunk in enumerate(relevant)
    ]
    system_prompt = _SYSTEM_TEMPLATE.format(context="\n\n".join(context_parts))
    return system_prompt, True
