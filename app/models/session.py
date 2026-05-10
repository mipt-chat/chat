"""
Модели для хранения истории диалогов.
"""

from datetime import datetime
from typing import List
from pydantic import BaseModel, Field
import uuid


class DialogMessage(BaseModel):
    """
    Одно сообщение в истории диалога.
    """
    message_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Уникальный идентификатор сообщения",
    )
    role: str = Field(
        ...,
        pattern=r"^(user|assistant|system)$",
        description="Роль отправителя: user, assistant или system",
        examples=["user", "assistant"],
    )
    content: str = Field(..., description="Текст сообщения")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Время создания сообщения",
    )


class Session(BaseModel):
    """
    Сессия пользователя с историей диалога.
    """
    session_id: str = Field(
        ...,
        description="Уникальный идентификатор сессии",
        examples=["tg_user_12345"],
    )
    messages: List[DialogMessage] = Field(
        default_factory=list,
        description="История сообщений в сессии",
    )
    updated_at: datetime = Field(
        default_factory=datetime.now,
        description="Время последнего обновления сессии",
    )