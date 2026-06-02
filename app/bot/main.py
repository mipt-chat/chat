"""
Точка входа Telegram-бота: `python -m app.bot.main`.

Соответствует CMD в Dockerfile.bot и команде сервиса `bot` в compose.yaml.
"""

from __future__ import annotations

import asyncio
import sys

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties

from app.bot.api_client import BackendAPIClient
from app.bot.handlers import build_router
from app.bot.session import SessionRegistry
from app.core import get_logger, setup_logging
from app.core.config import settings

logger = get_logger(__name__)


async def _run() -> None:
    if settings.telegram_bot_token is None:
        logger.error(
            "TELEGRAM_BOT_TOKEN is not set. Bot cannot start. "
            "Fill it in .env or pass via environment."
        )
        sys.exit(2)

    token = settings.telegram_bot_token.get_secret_value()
    bot = Bot(
        token=token,
        default=DefaultBotProperties(parse_mode=None),
    )
    sessions = SessionRegistry()

    async with BackendAPIClient(
        base_url=settings.backend_api_url,
        timeout_seconds=settings.llm_request_timeout_seconds,
    ) as api_client:
        dispatcher = Dispatcher()
        dispatcher.include_router(build_router(api_client, sessions))

        logger.info(
            "Starting Telegram bot polling. backend=%s streaming=%s throttle=%.2fs",
            settings.backend_api_url,
            settings.bot_streaming_enabled,
            settings.bot_draft_throttle_seconds,
        )

        try:
            await dispatcher.start_polling(bot, allowed_updates=dispatcher.resolve_used_update_types())
        finally:
            await bot.session.close()


def main() -> None:
    setup_logging()
    try:
        asyncio.run(_run())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot stopped")


if __name__ == "__main__":
    main()
