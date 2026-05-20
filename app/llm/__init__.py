"""
LLM Integration Layer.
Абстракция над GPT-провайдерами: YandexGPT, GigaChat, OpenAI-compatible.
"""

from app.llm.base import BaseLLMProvider
from app.llm.factory import get_llm_provider

__all__ = [
    "BaseLLMProvider",
    "get_llm_provider",
]
