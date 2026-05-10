"""
Сборщик метрик для unanswered вопросов и других показателей.

В MVP сохраняет метрики в JSON-файл. В будущем может быть заменён
на Prometheus, БД или интеграцию с CRM.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.core.logging_config import get_logger
from app.models.metrics import UnansweredQuery

logger = get_logger(__name__)

METRICS_DIR = Path("data/metrics")


class MetricsCollector:
    """
    Простой сборщик метрик на файлах.
    Использовать через глобальный экземпляр metrics_collector.
    """

    def __init__(self) -> None:
        self._metrics_dir = METRICS_DIR
        self._unanswered_file = self._metrics_dir / "unanswered_queries.jsonl"
        self._ensure_directory()

    def _ensure_directory(self) -> None:
        """Создаёт директорию для метрик, если её нет."""
        self._metrics_dir.mkdir(parents=True, exist_ok=True)

    def record_unanswered(
        self,
        question: str,
        session_id: Optional[str] = None,
        reason: str = "low_relevance",
    ) -> None:
        """
        Сохраняет вопрос, на который не нашлось ответа.

        Args:
            question: Исходный вопрос пользователя.
            session_id: Идентификатор сессии (если есть).
            reason: Причина отсутствия ответа:
                    - "low_relevance" — низкая релевантность чанков
                    - "missing_topic" — тема отсутствует в базе знаний
                    - "error" — ошибка при обработке запроса
        """
        record = UnansweredQuery(
            question=question,
            session_id=session_id,
            timestamp=datetime.now(),
            reason=reason,
        )

        try:
            with open(self._unanswered_file, "a", encoding="utf-8") as f:
                f.write(record.model_dump_json(ensure_ascii=False) + "\n")
            logger.info(
                f"Recorded unanswered query (reason={reason}): {question[:100]}..."
            )
        except OSError as e:
            logger.error(f"Failed to record unanswered query: {e}")

    def get_unanswered_count(self) -> int:
        """
        Возвращает количество записанных неотвеченных вопросов.

        Returns:
            Количество записей в файле unanswered_queries.jsonl.
        """
        if not self._unanswered_file.exists():
            return 0
        try:
            with open(self._unanswered_file, "r", encoding="utf-8") as f:
                return sum(1 for _ in f)
        except OSError as e:
            logger.error(f"Failed to read unanswered queries: {e}")
            return 0

    def get_unanswered_queries(self, limit: int = 100) -> list[UnansweredQuery]:
        """
        Возвращает последние N неотвеченных вопросов.

        Args:
            limit: Максимальное количество возвращаемых записей.

        Returns:
            Список объектов UnansweredQuery.
        """
        if not self._unanswered_file.exists():
            return []

        queries = []
        try:
            with open(self._unanswered_file, "r", encoding="utf-8") as f:
                # Читаем последние N строк
                lines = f.readlines()
                for line in lines[-limit:]:
                    if line.strip():
                        queries.append(UnansweredQuery.model_validate_json(line))
        except (OSError, json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to read unanswered queries: {e}")
        return queries

    def clear_unanswered(self) -> int:
        """
        Очищает файл неотвеченных вопросов.

        Returns:
            Количество удалённых записей.
        """
        count = self.get_unanswered_count()
        try:
            self._unanswered_file.write_text("", encoding="utf-8")
            logger.info(f"Cleared {count} unanswered queries")
        except OSError as e:
            logger.error(f"Failed to clear unanswered queries: {e}")
            return 0
        return count


# Глобальный экземпляр сборщика метрик
metrics_collector = MetricsCollector()