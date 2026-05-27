# Telegram-бот (Dev 5)

Документ для **Dev 5 (Telegram Bot)**: какой код добавлен в `app/bot/`, как он связан с остальными слоями и как его запускать/тестировать.

---

## Зона ответственности

| Команда | Зона |
|---------|------|
| **Dev 1** | `POST /chat`, `POST /chat/stream` (FastAPI). |
| **Dev 6** | `Settings`, `get_logger`, общие Pydantic-модели (`app/models/*`). |
| **Dev 5** | `app/bot/` — Telegram-клиент. Принимает сообщения, дёргает API, рисует ответ. |

Бот **ничего не знает** про базу знаний, ChromaDB, LLM-провайдеров и формирование промптов. Это сознательно: тогда RAG, LLM и retrieval можно менять без правок бота.

История диалога **хранится на бэкенде** (`app/core/session_store.py`). Бот лишь передаёт стабильный `session_id` и слушает handshake-поле `session_id` в ответе.

---

## Структура модуля

```
app/bot/
├── __init__.py
├── main.py          # entry point: `python -m app.bot.main`
├── api_client.py    # httpx-клиент к /chat и /chat/stream (SSE-парсер)
├── handlers.py      # /start, /help, /reset + обработчик текстовых сообщений
├── formatting.py    # split на ≤4096 символов, форматирование Источники
├── streaming.py     # «черновая» отрисовка через edit_message c throttle
└── session.py       # in-memory маппинг telegram user_id → session_id бэкенда

tests/bot/
├── test_api_client.py   # httpx.MockTransport, SSE-фикстуры
├── test_formatting.py
├── test_handlers.py     # фейк aiogram Message/Bot/Chat/User
└── test_session.py
```

---

## Используемые настройки (`Settings`)

Все переменные уже описаны в `app/core/config.py` и `.env.example` — бот не вводит **ни одной** новой переменной окружения.

| Переменная | Назначение |
|------------|-----------|
| `TELEGRAM_BOT_TOKEN` | Токен от @BotFather. Без него бот не стартует (exit 2). |
| `BACKEND_API_URL` | Базовый URL FastAPI-бэкенда. В Docker Compose — `http://api:8000`. |
| `BOT_STREAMING_ENABLED` | Включает «черновой» стриминг в приватных чатах. В группах всегда выключен. |
| `BOT_DRAFT_THROTTLE_SECONDS` | Минимальный интервал между `edit_text` в одном чате. |
| `BOT_MAX_FINAL_MESSAGE_CHARS` | Лимит длины одного сообщения Telegram (≤ 4096). |
| `FALLBACK_ANSWER` | Текст, если бэкенд не нашёл ответ в базе знаний. |
| `LLM_REQUEST_TIMEOUT_SECONDS` | Таймаут httpx-запросов к бэкенду. |

---

## Сценарии работы

### Приватный чат, `BOT_STREAMING_ENABLED=true`

1. Пользователь пишет сообщение.
2. Бот посылает `typing`, отправляет «…»-черновик и открывает `POST /chat/stream`.
3. По мере прихода `event: token` накапливает текст и не чаще чем раз в `throttle` секунд делает `edit_text`.
4. Если буфер не помещается в 4096 — фиксирует текущее сообщение и продолжает в новом.
5. На `event: done` записывает финальный текст (`answer` + блок «Источники»).
6. На `event: error` или сетевой обрыв — финализирует черновик вежливым сообщением об ошибке.

### Группа / `BOT_STREAMING_ENABLED=false`

1. Бот шлёт `typing`.
2. Делает синхронный `POST /chat`.
3. Получает `ChatResponse`, отправляет ответ (с разбивкой на части ≤ 4096, если нужно).

### Команды

- `/start` — приветствие, сбрасывает сессию (новый `session_id`).
- `/help` — список возможностей и команд.
- `/reset` — генерирует новый `session_id`, очищая историю на бэкенде de-facto.
- любое не-текстовое сообщение (фото, стикер, файл) — вежливый отказ.

---

## Контракт с API

Бот опирается **только** на:

- `app.models.chat.ChatRequest` / `ChatResponse` / `Source` — стабильные Pydantic-схемы из `app/models/`.
- SSE-формат `event: <name>\ndata: <json>\n\n`, события: `session`, `token`, `sources`, `done`, `error` (см. `_sse_event` в `app/services/chat_service.py`).

`session_id` формируется как `tg_user_{telegram_user_id}_{timestamp}_{rand}`. Бэкенд может вернуть свой `session_id` — бот его подхватывает (`SessionRegistry.update`).

---

## Запуск

### Локально (без Docker)

```bash
cp .env.example .env          # заполнить TELEGRAM_BOT_TOKEN и LLM-ключи
python -m app.bot.main
```

API при этом должен быть поднят отдельно (`python -m app.main`), а `BACKEND_API_URL` — указывать на него (по умолчанию `http://localhost:8000`).

### Через Docker Compose

```bash
docker compose --profile bot up -d api bot
```

Сервис `bot` в `compose.yaml` уже настроен: `depends_on: api healthy`, `BACKEND_API_URL=http://api:8000`, volume с кодом и логами.

---

## Тесты

```bash
pytest tests/bot/ -v
ruff check app/bot/ tests/bot/
```

Тесты бота не требуют тяжёлых зависимостей (`chromadb`, `sentence-transformers`, `torch`) — достаточно `pydantic`, `pydantic-settings`, `httpx`, `aiogram`, `pytest`, `pytest-asyncio`. Это даёт быстрый цикл разработки.

Внешние взаимодействия замоканы:

- `httpx.MockTransport` для бэкенда (включая SSE-поток);
- маленький `_FakeMessage`/`_FakeBot` для aiogram (без поднятия настоящего бота).

---

## Что **не** покрыто (вне MVP)

- Картинки/файлы от пользователя (бэкенд игнорирует `image`, бот выдаёт отказ).
- Inline-режим, кнопки, callback-и.
- Persistence сессий между перезапусками бота (на бэкенде история уже хранится — нужно лишь не терять `session_id`; можно добавить простой JSON-кэш позже).
- Webhook-режим (используется polling — этого достаточно для MVP).
