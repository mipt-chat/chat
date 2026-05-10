"""
Модели для API чата.
"""


from pydantic import BaseModel, Field


class Source(BaseModel):
    """
    Источник информации, использованный при генерации ответа.
    Ссылается на конкретный чанк из базы знаний.
    """
    chunk_id: str = Field(..., description="Уникальный идентификатор чанка")
    source_path: str = Field(
        ...,
        description="Относительный путь к исходному документу в базе знаний",
        examples=["instructions/returns/policy.txt"],
    )
    text: str = Field(..., description="Текст чанка, использованного как источник")
    score: float = Field(..., description="Оценка релевантности чанка (0.0 - 1.0)")


class ChatRequest(BaseModel):
    """
    Запрос от пользователя через API.
    session_id может представлять из себя как id пользователя в telegram,
    так и сессию анонимного пользователя в вебе
    """
    message: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Текст сообщения пользователя",
        examples=["Как оформить возврат товара?"],
    )
    session_id: str | None = Field(
        default=None,
        description="Идентификатор сессии для сохранения контекста диалога",
        examples=["tg_user_12345"],
    )
    image: str | None = Field(
        default=None,
        description="Скриншот в формате base64 (бонусная фича, необязательно)",
    )


class ChatResponse(BaseModel):
    """
    Ответ API на запрос пользователя.
    """
    answer: str = Field(..., description="Ответ, сгенерированный LLM")
    sources: list[Source] = Field(
        default_factory=list,
        description="Список источников, использованных при генерации ответа",
    )
    answered: bool = Field(
        default=True,
        description="Флаг: удалось ли найти релевантный ответ в базе знаний",
    )
    session_id: str | None = Field(
        default=None,
        description=(
            "Подтверждение идентификатора сессии. Если запрос без ID — "
            "возвращаем новый. Это handshake для новой сессии"
        ),
    )
