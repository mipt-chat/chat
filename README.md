# Customer Support AI Bot

MVP-проект чат-бота клиентской поддержки с использованием RAG-подхода.

## Быстрый старт

### Предварительные требования

- Python 3.10+
- pip или [uv](https://github.com/astral-sh/uv)
- Git

### Установка

1. Клонируйте репозиторий и перейдите в каталог проекта:

   ```bash
   git clone <repository-url>
   cd chat
   ```

2. Создайте и активируйте виртуальное окружение.

   Рекомендуется uv:

   ```bash
   uv venv
   source .venv/bin/activate   # Linux/macOS
   .venv\Scripts\activate      # Windows
   ```

   Классический вариант:

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/macOS
   venv\Scripts\activate      # Windows
   ```

3. Установите зависимости:

   ```bash
   uv pip install -r requirements.txt
   ```

   или:

   ```bash
   pip install -r requirements.txt
   ```

4. Переменные окружения:

   ```bash
   cp .env.example .env
   ```

   Отредактируйте `.env`: укажите реальные ключи API для выбранного LLM-провайдера. Файл `.env` в репозитории не коммитится (см. `.gitignore`).

   Важные переменные для локального запуска:

   | Переменная | Назначение |
   |------------|------------|
   | `KNOWLEDGE_BASE_FILE` | Точка входа в базу знаний: YAML или один `.txt` |
   | `EMBEDDING_MODEL_NAME` | Модель эмбеддингов для индексации (по умолчанию `intfloat/multilingual-e5-base`) |
   | `CHROMA_*` | Каталог и имя коллекции ChromaDB |

### Запуск через Docker Compose

После заполнения `.env` можно запускать без локального Python-окружения:

```bash
docker compose build
docker compose --profile tools run --rm indexer
docker compose up -d api
```

Проверки:

```bash
curl http://localhost:8000/health/live
curl http://localhost:8000/health/ready
```

Пример запроса:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Как зарегистрироваться физическому лицу?"}'
```

Встроенный веб-чат доступен на `http://localhost:8000/`.

Streaming API для веба и Telegram:

```bash
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message":"Как зарегистрироваться физическому лицу?","session_id":"web_demo"}'
```

Telegram-бот запускается отдельным сервисом после заполнения `TELEGRAM_BOT_TOKEN`:

```bash
docker compose --profile bot up -d api bot
```

Для личных чатов бот использует Telegram draft streaming, а финальный ответ
сохраняет обычным сообщением. Для групп и при ошибке draft-метода остаётся
обычный финальный ответ.

Быстрые тесты и lint в контейнере:

```bash
docker compose --profile tools run --rm test pytest tests/ -m "not slow" -q
docker compose --profile tools run --rm test ruff check app/ tests/
```

5. База знаний уже лежит в репозитории в каталоге `knowledge_base/`. Корневой манифест:

   - `knowledge_base/root.yaml` — список `imports` на разделы (`book/`, `business_requirements/`, `instructions/`, `law/`).

   В каждом разделе свой YAML с блоком `docs`: у каждой записи есть `location`, `type`, `source` (имя `.txt` файла). Тексты могут лежать **рядом с yaml** в той же папке (плоская раскладка) — пайплайн это поддерживает.

   Пример фрагмента `root.yaml`:

   ```yaml
   imports:
     - book/book.yaml
     - business_requirements/business_requirements.yaml
     - instructions/instructions.yaml
     - law/law.yaml
   ```

   Для одного большого файла без YAML можно задать `KNOWLEDGE_BASE_FILE=knowledge_base/knowledge.txt` и положить туда текст.

6. Индексация в ChromaDB (слой Data Processing, модель E5, префикс `passage:` для чанков):

   ```bash
   python -m app.data.indexing.pipeline
   ```

   Повторный запуск выполняет **инкрементальное** обновление: пересчитываются только изменённые или новые документы, удалённые из манифеста источники убираются из коллекции.

7. Запуск приложения:

   ```bash
   python -m app.main
   ```

## Структура проекта

Подробное описание структуры и правил разработки см. в [CONTRIBUTING.md](CONTRIBUTING.md).

Кратко по каталогам:

- `app/core/`, `app/models/` — конфигурация и общие модели данных
- `app/data/` — загрузка и индексация базы знаний (`app/data/indexing/pipeline.py`)
- `knowledge_base/` — YAML-манифесты и текстовые документы базы знаний
- `tests/` — тесты (в т.ч. `tests/data/` для Data Processing)

Интеграция **Retrieval (Dev 2)** с Chroma и эмбеддингами E5: [docs/DEV2_VECTOR_STORE.md](docs/DEV2_VECTOR_STORE.md).

## Технологический стек

- Backend: Python 3.10+, FastAPI, Uvicorn
- LLM: YandexGPT, GigaChat (OpenAI-совместимый интерфейс)
- Vector DB: ChromaDB
- Embeddings: sentence-transformers (`intfloat/multilingual-e5-base` по умолчанию)
- Telegram Bot: aiogram
- Testing: pytest

## Проверки перед коммитом

```bash
ruff check app/
pytest tests/
```

Полный `pytest tests/` включает помеченные `slow` тесты (загрузка модели эмбеддингов, проверка что `CHUNK_SIZE` укладывается в окно E5). Быстрее, без них:

```bash
pytest tests/ -m "not slow"
```

Явно только бюджет токенов под эмбеддер: `pytest tests/data/test_embedding_chunk_budget.py -v`.
