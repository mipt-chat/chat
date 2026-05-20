"""Chat endpoint."""

from fastapi import APIRouter

from app.models.chat import ChatRequest, ChatResponse
from app.services.chat_service import handle_chat_request

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Process a user chat message through Retrieval + LLM."""

    return await handle_chat_request(request)

