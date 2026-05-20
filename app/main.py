"""
Точка входа приложения.
Инициализирует логирование, конфигурацию и запускает FastAPI сервер.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from app.core.config import settings
from app.core.logging_config import get_logger, setup_logging

# Инициализация логирования до всего остального
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    logger.info(f"Starting {settings.app_name} v0.1.0")
    logger.info(f"Embedding model: {settings.embedding_model_name}")
    logger.info(f"Active LLM provider: {settings.active_llm_provider}")
    logger.info(f"ChromaDB collection: {settings.chroma_collection_name}")
    logger.info(f"RAG top-k: {settings.retrieval_top_k}")
    yield
    logger.info(f"Shutting down {settings.app_name}")


def create_app() -> FastAPI:
    """
    Создаёт и настраивает FastAPI приложение.
    Выделено в отдельную функцию для тестирования.
    """
    from app.api.errors import register_exception_handlers
    from app.api.routes import chat, health, web

    # Импорты здесь, чтобы логирование было уже настроено

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="MVP-система чат-бота клиентской поддержки с RAG",
        lifespan=lifespan,
    )

    register_exception_handlers(app)
    app.include_router(web.router)
    app.include_router(chat.router)
    app.include_router(health.router)

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
    )
