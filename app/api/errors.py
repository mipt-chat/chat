"""FastAPI exception handlers for the public API."""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.core import get_logger
from app.core.exceptions import (
    AppException,
    ConfigurationError,
    EmbeddingError,
    KnowledgeBaseNotFoundError,
    LLMProviderError,
    RetrievalError,
)
from app.models.error import ErrorResponse

logger = get_logger(__name__)


def _error_response(status_code: int, error: str, detail: str | None = None) -> JSONResponse:
    payload = ErrorResponse(error=error, detail=detail)
    return JSONResponse(status_code=status_code, content=payload.model_dump())


async def retrieval_exception_handler(
    request: Request,
    exc: RetrievalError | EmbeddingError,
) -> JSONResponse:
    logger.warning("Retrieval failure on %s %s: %s", request.method, request.url.path, exc.message)
    return _error_response(
        status.HTTP_503_SERVICE_UNAVAILABLE,
        exc.message,
        exc.detail,
    )


async def llm_exception_handler(request: Request, exc: LLMProviderError) -> JSONResponse:
    logger.warning("LLM failure on %s %s: %s", request.method, request.url.path, exc.message)
    return _error_response(
        status.HTTP_502_BAD_GATEWAY,
        exc.message,
        exc.detail,
    )


async def configuration_exception_handler(
    request: Request,
    exc: ConfigurationError | KnowledgeBaseNotFoundError,
) -> JSONResponse:
    logger.error("Configuration failure on %s %s: %s", request.method, request.url.path, exc.message)
    return _error_response(
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        exc.message,
        exc.detail,
    )


async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    logger.error("Application failure on %s %s: %s", request.method, request.url.path, exc.message)
    return _error_response(
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        exc.message,
        exc.detail,
    )


async def unexpected_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unexpected failure on %s %s", request.method, request.url.path)
    return _error_response(
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        "Internal server error",
        str(exc),
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Attach public API error handlers to a FastAPI application."""

    app.add_exception_handler(RetrievalError, retrieval_exception_handler)
    app.add_exception_handler(EmbeddingError, retrieval_exception_handler)
    app.add_exception_handler(LLMProviderError, llm_exception_handler)
    app.add_exception_handler(ConfigurationError, configuration_exception_handler)
    app.add_exception_handler(KnowledgeBaseNotFoundError, configuration_exception_handler)
    app.add_exception_handler(AppException, app_exception_handler)
    app.add_exception_handler(Exception, unexpected_exception_handler)
