"""
Базовые исключения приложения.
Все кастомные исключения должны наследоваться от AppException.
"""


class AppException(Exception):
    """Базовое исключение приложения."""

    def __init__(self, message: str, detail: str | None = None):
        self.message = message
        self.detail = detail
        super().__init__(message)


class ConfigurationError(AppException):
    """Ошибка конфигурации (неверные переменные окружения, отсутствие файлов)."""
    pass


class KnowledgeBaseNotFoundError(AppException):
    """Файл базы знаний не найден."""
    pass


class EmbeddingError(AppException):
    """Ошибка при создании эмбеддингов."""
    pass


class RetrievalError(AppException):
    """Ошибка при поиске в векторной базе."""
    pass


class LLMProviderError(AppException):
    """Ошибка при взаимодействии с LLM-провайдером."""
    pass


class SessionError(AppException):
    """Ошибка при работе с сессиями."""
    pass
