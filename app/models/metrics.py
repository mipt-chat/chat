"""
Модели для сбора метрик.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class UnansweredQuery(BaseModel):
    """
    Запись о вопросе, на который не нашлось ответа в базе знаний.
    Сохраняется для последующей интеграции с CRM.
    """
    question: str = Field(..., description="Исходный вопрос пользователя")
    session_id: Optional[str] = Field(
        default=None,
        description="Идентификатор сессии пользователя",
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Время получения вопроса без ответа",
    )
    reason: str = Field(
        default="low_relevance",
        description="Причина отсутствия ответа (low_relevance, missing_topic, error)",
        examples=["low_relevance", "missing_topic", "error"],
    )