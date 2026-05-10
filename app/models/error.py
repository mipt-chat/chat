"""
Модели ошибок API.
"""

from typing import Optional
from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """
    Стандартный ответ API при возникновении ошибки.
    """
    error: str = Field(
        ...,
        description="Краткое описание ошибки",
        examples=["Knowledge base not found", "LLM provider timeout"],
    )
    detail: Optional[str] = Field(
        default=None,
        description="Подробное описание ошибки для разработчика",
    )