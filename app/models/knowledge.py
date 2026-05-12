"""
Модели для работы с базой знаний и retrieval.
"""

from typing import Any

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    """
    Чанк, найденный в результате similarity search.
    Передаётся из RAG-слоя в LLM-слой и API.
    """
    chunk_id: str = Field(..., description="Уникальный ID чанка в ChromaDB")
    text: str = Field(..., description="Текстовое содержимое чанка")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Метаданные чанка. Ключ 'source_path' обязателен для базы знаний",
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Оценка релевантности (косинусное расстояние)",
    )


class KnowledgeDocument(BaseModel):
    """
    Документ базы знаний до разбиения на чанки.
    Используется в слое data processing.
    """
    source_path: str = Field(
        ...,
        description="Относительный путь к документу в базе знаний",
        examples=["instructions/returns/policy.txt"],
    )
    content: str = Field(..., description="Полное содержимое документа")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Дополнительные метаданные документа",
    )


class IndexedChunk(BaseModel):
    """
    Чанк после нарезки документа: готов к эмбеддингу и записи в ChromaDB.
    Общая модель слоя data — в хранилище передаются поля ``chunk_id``, ``text``, ``metadata``.
    """

    chunk_id: str = Field(
        ...,
        description="Уникальный идентификатор записи в ChromaDB (обычно source_path:chunk_index)",
    )
    text: str = Field(..., description="Текст чанка без префикса passage:")
    chunk_index: int = Field(..., ge=0, description="Порядковый номер чанка внутри исходного документа")
    metadata: dict[str, str | int | float | bool] = Field(
        default_factory=dict,
        description="Метаданные строки в ChromaDB (без null; source_path, doc_hash, chunk_index, …)",
    )
