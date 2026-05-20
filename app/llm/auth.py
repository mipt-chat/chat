"""Authentication strategies for OpenAI-compatible LLM providers."""

import time
from typing import Protocol
from uuid import uuid4

import httpx

from app.core import get_logger
from app.core.exceptions import LLMProviderError

logger = get_logger(__name__)


class LLMAuthProvider(Protocol):
    """Returns the API key/token that should be passed to OpenAI SDK."""

    async def get_api_key(self) -> str:
        """Return a ready-to-use API key or bearer token."""
        ...


class StaticApiKeyAuth:
    """Static API key auth for regular OpenAI-compatible endpoints."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key or "no-key"

    async def get_api_key(self) -> str:
        return self._api_key


class GigaChatOAuthAuth:
    """GigaChat credentials-to-access-token exchange."""

    def __init__(
        self,
        *,
        credentials: str,
        auth_url: str,
        scope: str,
        verify_ssl: bool,
    ) -> None:
        self._credentials = credentials
        self._auth_url = auth_url
        self._scope = scope
        self._verify_ssl = verify_ssl
        self._access_token: str | None = None
        self._expires_at = 0.0

    async def get_api_key(self) -> str:
        now = time.time()
        if self._access_token and self._expires_at > now + 60:
            return self._access_token

        if not self._auth_url:
            raise LLMProviderError(
                message="Не настроен OAuth URL для GigaChat",
                detail="Set GIGACHAT_AUTH_URL or disable GIGACHAT_USE_OAUTH.",
            )

        logger.info("Refreshing GigaChat OAuth access token")
        try:
            async with httpx.AsyncClient(verify=self._verify_ssl, timeout=30.0) as client:
                response = await client.post(
                    self._auth_url,
                    headers={
                        "Authorization": self._authorization_header(),
                        "RqUID": str(uuid4()),
                        "Accept": "application/json",
                        "Content-Type": "application/x-www-form-urlencoded",
                    },
                    data={"scope": self._scope},
                )
                response.raise_for_status()
                payload = response.json()
        except Exception as exc:
            raise LLMProviderError(
                message="Ошибка при получении OAuth-токена GigaChat",
                detail=str(exc),
            ) from exc

        access_token = payload.get("access_token")
        if not access_token:
            raise LLMProviderError(
                message="Ошибка при получении OAuth-токена GigaChat",
                detail="OAuth response has no access_token",
            )

        self._access_token = str(access_token)
        expires_at = payload.get("expires_at")
        if isinstance(expires_at, int | float):
            # GigaChat returns Unix timestamp in milliseconds.
            self._expires_at = (
                float(expires_at) / 1000 if expires_at > 10_000_000_000 else float(expires_at)
            )
        else:
            self._expires_at = now + 25 * 60

        logger.info("GigaChat OAuth access token refreshed")
        return self._access_token

    def _authorization_header(self) -> str:
        credentials = self._credentials.strip()
        if credentials.lower().startswith("basic "):
            return credentials
        return f"Basic {credentials}"
