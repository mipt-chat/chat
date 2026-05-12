"""
Централизованная конфигурация приложения.
Единственный источник правды для всех настроек.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Главный класс конфигурации приложения."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ============================================================
    # Общие
    # ============================================================
    app_name: str = "CustomerSupportBot"
    debug: bool = False

    # ============================================================
    # База знаний
    # ============================================================
    knowledge_base_file: Path = Path("knowledge_base/root.yaml")

    # ============================================================
    # ChromaDB
    # ============================================================
    chroma_persist_directory: str = "chroma_storage"
    chroma_collection_name: str = "support_knowledge"

    # ============================================================
    # Embeddings
    # ============================================================
    embedding_model_name: str = "intfloat/multilingual-e5-base"

    # ============================================================
    # RAG параметры
    # ============================================================
    retrieval_top_k: int = Field(default=5, ge=1, le=20)
    chunk_size: int = Field(default=1000, ge=100, le=5000)
    chunk_overlap: int = Field(default=200, ge=0, le=1000)

    # ============================================================
    # LLM Провайдеры
    # ============================================================
    active_llm_provider: Literal["yandex", "giga", "openai_compatible"] = "yandex"

    # YandexGPT
    yandex_api_key: SecretStr | None = None
    yandex_folder_id: str | None = None
    yandex_base_url: str = "https://llm.api.cloud.yandex.net/foundationModels/v1/completions"
    yandex_model_name: str = "yandexgpt"

    # GigaChat
    gigachat_api_key: SecretStr | None = None
    gigachat_base_url: str = "https://gigachat.devices.sberbank.ru/api/v1"
    gigachat_model_name: str = "gigachat"

    # Универсальный OpenAI-совместимый
    openai_compatible_api_key: SecretStr | None = None
    openai_compatible_base_url: str = "http://localhost:11434/v1"
    openai_compatible_model_name: str = "llama3"

    def get_active_provider_config(self) -> dict:
        """
        Возвращает конфигурацию активного провайдера в виде словаря.
        Используется для инициализации LLM-клиента в слое llm/.
        """
        provider_configs = {
            "yandex": {
                "api_key": (
                    self.yandex_api_key.get_secret_value()
                    if self.yandex_api_key
                    else None
                ),
                "base_url": self.yandex_base_url,
                "folder_id": self.yandex_folder_id,
                "model_name": "yandexgpt",
            },
            "giga": {
                "api_key": (
                    self.gigachat_api_key.get_secret_value()
                    if self.gigachat_api_key
                    else None
                ),
                "base_url": self.gigachat_base_url,
                "model_name": "gigachat",
            },
            "openai_compatible": {
                "api_key": (
                    self.openai_compatible_api_key.get_secret_value()
                    if self.openai_compatible_api_key
                    else None
                ),
                "base_url": self.openai_compatible_base_url,
                "model_name": self.openai_compatible_model_name,
            },
        }
        return provider_configs[self.active_llm_provider]

    # ============================================================
    # Fallback
    # ============================================================
    fallback_answer: str = (
        "Извините, я не нашёл ответ на ваш вопрос в базе знаний. "
        "Пожалуйста, уточните запрос или обратитесь к оператору поддержки."
    )

    # ============================================================
    # История диалогов
    # ============================================================
    max_history_length: int = Field(default=5, ge=0, le=20)
    session_ttl_days: int = Field(default=30, ge=0)

    # ============================================================
    # Логирование
    # ============================================================
    log_level_console: str = "DEBUG"
    log_level_file: str = "INFO"
    log_dir: Path = Path("logs")
    log_file: str = "bot.log"

# Глобальный синглтон конфигурации — единственный экземпляр на всё приложение
settings = Settings()
