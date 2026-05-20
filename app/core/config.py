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
    yandex_base_url: str = "https://llm.api.cloud.yandex.net/v1"
    yandex_model_name: str = "yandexgpt"

    # GigaChat
    gigachat_api_key: SecretStr | None = None
    gigachat_base_url: str = "https://gigachat.devices.sberbank.ru/api/v1"
    gigachat_model_name: str = "GigaChat"
    gigachat_auth_url: str = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    gigachat_scope: str = "GIGACHAT_API_PERS"
    gigachat_use_oauth: bool = True
    gigachat_verify_ssl: bool = False

    # Универсальный OpenAI-совместимый
    openai_compatible_api_key: SecretStr | None = None
    openai_compatible_base_url: str = "http://localhost:11434/v1"
    openai_compatible_model_name: str = "llama3"

    @staticmethod
    def _secret_value(value: SecretStr | str | None) -> str | None:
        if value is None:
            return None
        if hasattr(value, "get_secret_value"):
            return value.get_secret_value()
        return str(value)

    def get_active_provider_config(self) -> dict:
        """
        Возвращает конфигурацию активного провайдера в виде словаря.
        Используется для инициализации LLM-клиента в слое llm/.
        """
        provider_configs = {
            "yandex": {
                "api_key": self._secret_value(self.yandex_api_key),
                "base_url": self.yandex_base_url,
                "folder_id": self.yandex_folder_id,
                "model_name": self.yandex_model_name,
            },
            "giga": {
                "api_key": self._secret_value(self.gigachat_api_key),
                "base_url": self.gigachat_base_url,
                "model_name": self.gigachat_model_name,
                "auth_url": self.gigachat_auth_url,
                "scope": self.gigachat_scope,
                "use_oauth": self.gigachat_use_oauth,
                "verify_ssl": self.gigachat_verify_ssl,
            },
            "openai_compatible": {
                "api_key": self._secret_value(self.openai_compatible_api_key),
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
    # Telegram Bot
    # ============================================================
    telegram_bot_token: SecretStr | None = None
    backend_api_url: str = "http://localhost:8000"
    bot_streaming_enabled: bool = True
    bot_draft_throttle_seconds: float = Field(default=1.0, ge=0.2, le=5.0)
    bot_max_final_message_chars: int = Field(default=4096, ge=1000, le=4096)

    # ============================================================
    # Логирование
    # ============================================================
    log_level_console: str = "DEBUG"
    log_level_file: str = "INFO"
    log_dir: Path = Path("logs")
    log_file: str = "bot.log"

# Глобальный синглтон конфигурации — единственный экземпляр на всё приложение
settings = Settings()
