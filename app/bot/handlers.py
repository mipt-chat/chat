"""
Хендлеры aiogram: команды и обработка сообщений пользователя.

Команды:
- /start  — приветствие, инициализирует сессию.
- /help   — краткая справка по возможностям.
- /reset  — сбрасывает контекст диалога на стороне бэкенда.

Любой текст — уходит в API. Не-текст — вежливый отказ.
"""

from __future__ import annotations

from aiogram import F, Router
from aiogram.enums import ChatAction, ChatType
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

from app.bot.api_client import BackendAPIClient, BackendAPIError, parse_sources
from app.bot.formatting import (
    TELEGRAM_MAX_MESSAGE_CHARS,
    compose_final_message,
    split_for_telegram,
)
from app.bot.session import SessionRegistry
from app.bot.streaming import StreamingMessageRenderer
from app.core import get_logger
from app.core.config import settings
from app.models.chat import Source

logger = get_logger(__name__)

WELCOME_TEXT = (
    "Привет! Я бот клиентской поддержки Токеон. "
    "Отвечаю на вопросы о цифровых финансовых активах (ЦФА), "
    "регистрации, документах и работе платформы.\n\n"
    "Просто напишите свой вопрос. Команды:\n"
    "/help — что я умею\n"
    "/reset — начать новый диалог"
)

HELP_TEXT = (
    "Я отвечаю на вопросы о ЦФА на базе официальной документации Токеон.\n\n"
    "Что важно знать:\n"
    "• помню последние 5 сообщений нашей беседы;\n"
    "• отвечаю только по базе знаний, не ищу в интернете;\n"
    "• если не нашёл ответ — честно скажу об этом;\n"
    "• картинки и файлы пока не понимаю — только текст.\n\n"
    "Команды:\n"
    "/start — приветствие\n"
    "/reset — забыть прошлый диалог и начать заново"
)

RESET_TEXT = (
    "Готово, начинаем новый диалог. О чём хотите спросить?"
)

NON_TEXT_REPLY = (
    "Я пока понимаю только текстовые сообщения. "
    "Опишите вопрос словами, пожалуйста."
)

ERROR_REPLY = (
    "Не удалось получить ответ от сервиса. Попробуйте, пожалуйста, ещё раз чуть позже."
)


def build_router(
    api_client: BackendAPIClient,
    sessions: SessionRegistry,
) -> Router:
    """
    Собирает aiogram-роутер с привязанными зависимостями.

    Зависимости передаются явно (а не через DI aiogram), чтобы упростить
    юнит-тестирование хендлеров.
    """
    router = Router(name="customer-support-bot")

    @router.message(CommandStart())
    async def on_start(message: Message) -> None:
        if message.from_user is not None:
            sessions.reset(message.from_user.id)
        await message.answer(WELCOME_TEXT)

    @router.message(Command("help"))
    async def on_help(message: Message) -> None:
        await message.answer(HELP_TEXT)

    @router.message(Command("reset"))
    async def on_reset(message: Message) -> None:
        if message.from_user is not None:
            sessions.reset(message.from_user.id)
        await message.answer(RESET_TEXT)

    @router.message(F.text)
    async def on_text(message: Message) -> None:
        await _handle_user_question(message, api_client, sessions)

    @router.message()
    async def on_unsupported(message: Message) -> None:
        await message.answer(NON_TEXT_REPLY)

    return router


async def _handle_user_question(
    message: Message,
    api_client: BackendAPIClient,
    sessions: SessionRegistry,
) -> None:
    user = message.from_user
    text = (message.text or "").strip()
    if not text or user is None:
        return

    session_id = sessions.get_or_create(user.id)
    is_private = message.chat.type == ChatType.PRIVATE
    use_streaming = settings.bot_streaming_enabled and is_private

    try:
        await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)
    except Exception:  # noqa: BLE001 — typing action не критичен
        logger.debug("send_chat_action failed; continuing")

    try:
        if use_streaming:
            await _reply_with_stream(message, api_client, sessions, user.id, text, session_id)
        else:
            await _reply_plain(message, api_client, sessions, user.id, text, session_id)
    except BackendAPIError as exc:
        logger.warning("Backend API error for user %s: %s", user.id, exc)
        await message.answer(ERROR_REPLY)
    except Exception:
        logger.exception("Unexpected error while handling message from user %s", user.id)
        await message.answer(ERROR_REPLY)


async def _reply_plain(
    message: Message,
    api_client: BackendAPIClient,
    sessions: SessionRegistry,
    user_id: int,
    text: str,
    session_id: str,
) -> None:
    response = await api_client.chat(message=text, session_id=session_id)
    if response.session_id:
        sessions.update(user_id, response.session_id)

    sources = response.sources if response.answered else []
    final_text = compose_final_message(response.answer, sources)
    await _send_long(message, final_text)


async def _reply_with_stream(
    message: Message,
    api_client: BackendAPIClient,
    sessions: SessionRegistry,
    user_id: int,
    text: str,
    session_id: str,
) -> None:
    draft = await message.answer("…")

    async def send_new(part: str) -> Message:
        return await message.answer(part or "…")

    renderer = StreamingMessageRenderer(
        draft_message=draft,
        send_new=send_new,
        throttle_seconds=settings.bot_draft_throttle_seconds,
        message_char_limit=settings.bot_max_final_message_chars,
    )

    final_answer: str | None = None
    final_answered: bool = True
    final_sources: list[Source] = []
    backend_session_id: str | None = None

    try:
        async for event in api_client.stream_chat(message=text, session_id=session_id):
            if event.type == "session":
                backend_session_id = event.data.get("session_id")
            elif event.type == "token":
                token = event.data.get("text") or ""
                await renderer.push(token)
            elif event.type == "sources":
                final_sources = parse_sources(event.data.get("sources"))
            elif event.type == "done":
                final_answer = event.data.get("answer")
                final_answered = bool(event.data.get("answered", True))
                final_sources = parse_sources(event.data.get("sources")) or final_sources
                if not backend_session_id:
                    backend_session_id = event.data.get("session_id")
            elif event.type == "error":
                raise BackendAPIError(
                    str(event.data.get("error") or "stream error")
                )
    except BackendAPIError:
        await renderer.finalize(ERROR_REPLY)
        raise

    if backend_session_id:
        sessions.update(user_id, backend_session_id)

    final_text = compose_final_message(
        final_answer or settings.fallback_answer,
        final_sources if final_answered else [],
    )
    await renderer.finalize(final_text)


async def _send_long(message: Message, text: str) -> None:
    """Отправляет текст одним или несколькими сообщениями ≤ 4096 символов."""
    parts = split_for_telegram(text, limit=TELEGRAM_MAX_MESSAGE_CHARS)
    if not parts:
        await message.answer(settings.fallback_answer)
        return
    for part in parts:
        await message.answer(part)
