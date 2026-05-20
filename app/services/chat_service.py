"""Orchestration service for the chat API."""

from pathlib import Path
from uuid import uuid4

from fastapi.concurrency import run_in_threadpool

from app.core import get_logger, session_store
from app.llm import get_llm_provider
from app.metrics.collector import metrics_collector
from app.models.chat import ChatRequest, ChatResponse, Source
from app.models.knowledge import RetrievedChunk
from app.rag import search_context

logger = get_logger(__name__)


def _ensure_session_id(session_id: str | None) -> str:
    normalized = session_id.strip() if session_id else ""
    return normalized or f"web_{uuid4()}"


def _source_path_for_api(raw_source_path: object) -> str:
    if not raw_source_path:
        return ""

    source_path = str(raw_source_path)
    try:
        path = Path(source_path)
        if path.is_absolute():
            return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
        return path.as_posix()
    except ValueError:
        return source_path.replace("\\", "/")
    except OSError:
        return source_path.replace("\\", "/")


def _chunk_to_source(chunk: RetrievedChunk) -> Source:
    return Source(
        chunk_id=chunk.chunk_id,
        source_path=_source_path_for_api(chunk.metadata.get("source_path")),
        text=chunk.text,
        score=chunk.score,
    )


async def handle_chat_request(request: ChatRequest) -> ChatResponse:
    """Run the full RAG + LLM chat pipeline for one user message."""

    session_id = _ensure_session_id(request.session_id)
    message = request.message.strip()

    if request.image:
        logger.info("ChatRequest.image was provided but image processing is not implemented in MVP")

    history = session_store.get_recent_messages(session_id)
    chunks = await run_in_threadpool(search_context, message)

    provider = get_llm_provider()
    answer, answered = await provider.generate(
        question=message,
        context_chunks=chunks,
        history=history,
    )

    session_store.add_message(session_id, "user", message)
    session_store.add_message(session_id, "assistant", answer)

    if not answered:
        metrics_collector.record_unanswered(
            question=message,
            session_id=session_id,
            reason="low_relevance",
        )

    logger.info(
        "Chat request processed: session_id=%s answered=%s chunks=%d",
        session_id,
        answered,
        len(chunks),
    )

    return ChatResponse(
        answer=answer,
        sources=[_chunk_to_source(chunk) for chunk in chunks],
        answered=answered,
        session_id=session_id,
    )
