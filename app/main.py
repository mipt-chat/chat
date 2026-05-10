"""
Точка входа приложения.
Инициализирует логирование, конфигурацию и запускает FastAPI сервер.
"""

import uvicorn
from app.core.logging_config import setup_logging, get_logger
from app.core.config import settings

# Инициализация логирования до всего остального
setup_logging()
logger = get_logger(__name__)


def create_app():
    """
    Создаёт и настраивает FastAPI приложение.
    Выделено в отдельную функцию для тестирования.
    """
    from fastapi import FastAPI
    # Импорты здесь, чтобы логирование было уже настроено

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="MVP-система чат-бота клиентской поддержки с RAG",
    )

    # Регистрация роутов будет добавлена Developer 1
    # from app.api.routes import chat, health
    # app.include_router(chat.router)
    # app.include_router(health.router)

    return app


app = create_app()


@app.on_event("startup")
async def startup_event():
    logger.info(f"Starting {settings.app_name} v0.1.0")
    logger.info(f"Embedding model: {settings.embedding_model_name}")
    logger.info(f"Active LLM provider: {settings.active_llm_provider}")
    logger.info(f"ChromaDB collection: {settings.chroma_collection_name}")
    logger.info(f"RAG top-k: {settings.retrieval_top_k}")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info(f"Shutting down {settings.app_name}")


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
    )