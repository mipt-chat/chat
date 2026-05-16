"""
Единая конфигурация логирования для всего приложения.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.core.config import settings

# Флаг, чтобы избежать повторной инициализации
_logging_initialized: bool = False


def setup_logging() -> None:
    """
    Инициализирует логирование для всего приложения.
    Должна вызываться один раз при старте в main.py.
    """
    global _logging_initialized
    if _logging_initialized:
        return

    # Уровни логирования из конфига
    console_level = getattr(logging, settings.log_level_console.upper(), logging.DEBUG)
    file_level = getattr(logging, settings.log_level_file.upper(), logging.INFO)

    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Создаём директорию для логов
    log_dir = Path(settings.log_dir)
    log_dir.mkdir(exist_ok=True)

    # Получаем корневой логгер
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Корень принимает всё, фильтруют handler-ы

    # Очищаем существующие handler-ы
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # --- Console Handler (DEBUG) ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    root_logger.addHandler(console_handler)

    # --- File Handler (INFO и выше) ---
    file_handler = RotatingFileHandler(
        log_dir / settings.log_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    root_logger.addHandler(file_handler)

    # Делаем сторонние библиотеки менее шумными
    logging.getLogger("httpx").setLevel(logging.WARNING)
    # Иначе при завершении pytest (stdout уже закрыт) HF/httpx могут дать DEBUG из
    # httpcore → "I/O operation on closed file"
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("aiogram").setLevel(logging.WARNING)

    _logging_initialized = True

    # Первое сообщение в лог
    logger = logging.getLogger(__name__)
    logger.info("Logging initialized")


def get_logger(name: str) -> logging.Logger:
    """
    Возвращает настроенный логгер с указанным именем.
    Единственный способ получения логгера в приложении.

    Args:
        name: Обычно передавать __name__ из вызывающего модуля.

    Returns:
        Настроенный экземпляр logging.Logger.
    """
    if not _logging_initialized:
        setup_logging()
    return logging.getLogger(name)
