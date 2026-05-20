"""
LLM Integration Layer.
Абстракция над GPT-провайдерами: YandexGPT, GigaChat, OpenAI-compatible.
"""

from app.llm.base import BaseLLMProvider
from app.llm.factory import close_llm_provider, get_llm_provider

__all__ = [
    "BaseLLMProvider",
    "close_llm_provider",
    "get_llm_provider",
]
