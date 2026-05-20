"""Health endpoints for local development and Docker."""

from pathlib import Path

from fastapi import APIRouter, HTTPException, status

from app.core.config import settings

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live")
async def live() -> dict[str, str]:
    """Liveness check: the process can accept HTTP requests."""

    return {"status": "ok"}


@router.get("/ready")
async def ready() -> dict[str, str | int]:
    """Readiness check: Chroma index exists and contains chunks."""

    chroma_dir = Path(settings.chroma_persist_directory)
    if not chroma_dir.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "ChromaDB directory is missing. Run indexing first: "
                "python -m app.data.indexing.pipeline"
            ),
        )

    try:
        import chromadb

        client = chromadb.PersistentClient(path=settings.chroma_persist_directory)
        collection = client.get_collection(settings.chroma_collection_name)
        chunk_count = collection.count()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "ChromaDB collection is not ready. Run indexing first: "
                "python -m app.data.indexing.pipeline"
            ),
        ) from exc

    if chunk_count <= 0:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "ChromaDB collection is empty. Run indexing first: "
                "python -m app.data.indexing.pipeline"
            ),
        )

    provider_config = settings.get_active_provider_config()
    api_key = provider_config.get("api_key")
    if not api_key or str(api_key) == "no-key":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM provider '{settings.active_llm_provider}' has no API key configured.",
        )

    return {
        "status": "ready",
        "collection": settings.chroma_collection_name,
        "chunks": chunk_count,
        "llm_provider": settings.active_llm_provider,
    }

