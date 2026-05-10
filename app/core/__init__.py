"""
Core инфраструктура приложения: конфигурация, логирование, сессии.
"""

from app.core.logging_config import get_logger, setup_logging

__all__ = [
    "get_logger",
    "setup_logging",
]