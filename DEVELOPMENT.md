# Техническое задание для разработчиков

Актуально на: 10.05.2026

## 1. Что уже готово (Shared Infrastructure)

### 1.1. Конфигурация (`app/core/config.py`)
- Все настройки в классе `Settings` (pydantic-settings)
- Переменные окружения загружаются из `.env`
- **Секреты (ключи API) использовать только через `SecretStr`** — они не светятся в логах
- Добавление новых переменных: сначала в `Settings`, потом в `.env.example`
- Получение конфига в любом компоненте:
  ```python
  from app.core.config import settings
  print(settings.retrieval_top_k)
  ```
  
### 1.2 Логирование (app/core/logging_config.py)

- Только через get_logger(__name__) — никаких print() и logging.getLogger() напрямую
- Уровни:
  - DEBUG — только в консоли
  - INFO и выше — в консоли и в файле logs/bot.log

Не менять глобальные настройки логирования в своих компонентах

### 1.3 Модели данных (app/models/)

| Модель              | Файл            | Назначение                            |
|:--------------------|:---------------:|:--------------------------------------|
| `ChatRequest`       | `chat.py`       | Запрос от пользователя                |
| `ChatResponse`      | `chat.py`       | Ответ пользователю                    |
| `Source`            | `chat.py`       | Источник (чанк) в ответе              |
| `RetrievedChunk`    | `knowledge.py`  | Чанк из RAG                           |
| `KnowledgeDocument` | `knowledge.py`  | Документ базы знаний                  |
| `Session`           | `session.py`    | Сессия с историей диалога             |
| `DialogMessage`     | `session.py`    | Одно сообщение в сессии               |
| `UnansweredQuery`   | `metrics.py`    | Вопрос без ответа                     |
| `ErrorResponse`     | `error.py`      | Ошибка API                            |
Правило: если данные идут из одного компонента в другой — они должны быть экземпляром одной из этих моделей. Локальные модели внутри компонента — можно, но наружу не отдавать.

### 1.4. Сессии (app/core/session_store.py)

- Хранение истории диалогов — JSON-файлы в data/sessions/
- Не использовать БД для сессий в MVP
- Максимум 5 диалогов (10 сообщений) в сессии — настраивается в .env (MAX_HISTORY_LENGTH)
- Сессии старше 30 дней автоматически удаляются (SESSION_TTL_DAYS)
- Интерфейс: load_session(), save_session(), add_message(), get_recent_messages()
- В будущем возможна замена на PostgreSQL через тот же интерфейс (BaseSessionStore)

### 1.5. Метрики (app/metrics/collector.py)

- Сохранение неотвеченных вопросов в data/metrics/unanswered_queries.jsonl
- через metrics_collector.record_unanswered(question, session_id, reason)
- Причины: "low_relevance", "missing_topic", "error"

### 1.6. Исключения (app/core/exceptions.py)

- Базовое: AppException
- Специфичные: ConfigurationError, KnowledgeBaseNotFoundError, EmbeddingError, RetrievalError, LLMProviderError, SessionError

Наследуйте свои исключения от AppException

## 2. Ограничения MVP
Не делаем в MVP:
- Реляционная БД (PostgreSQL, SQLite)
- Хранение сессий в БД — только JSON
- Обработка скриншотов (поле image в ChatRequest есть, но не реализовано)
- Интеграция с CRM
- Поиск в интернете — только база знаний
- Вложения (файлы)
- Сложная аналитика

Делаем в MVP:
- Telegram-бот с текстовыми сообщениями
- RAG-поиск по базе знаний (ChromaDB)
- LLM-ответы через YandexGPT / GigaChat / OpenAI-совместимые
- История диалогов (5 последних сообщений)
- Вежливый fallback при отсутствии ответа
- Сбор метрик по неотвеченным вопросам

## 3. База знаний
Структура knowledge_base:
```commandline
kb/
├── book/
├── business_requirements/
├── instructions/
└── law/
    └── documents/
```

- ~300 КБ, текстовая, статичная
- Обновляется 3-5 раз в месяц

При индексации сохранять source_path относительно knowledge_base/ в метаданные чанка.
Пример source_path: "law/documents/law115.txt"

4. API-контракт (общий)
POST /chat (будет реализован Dev 1)
json
```json
// Request
{
  "message": "Как оформить возврат товара?",
  "session_id": "tg_user_12345",
  "image": null
}

// Response
{
  "answer": "Для возврата нужно...",
  "sources": [
    {
      "chunk_id": "chunk_001",
      "source_path": "instructions/returns/policy.txt",
      "text": "Для возврата товара необходимо...",
      "score": 0.92
    }
  ],
  "answered": true,
  "session_id": "tg_user_12345"
}
```


5. Куда смотреть

| Разработчик | Компонент              | Пакет                              |
|:-----------:|:-----------------------|:-----------------------------------|
|   **Dev 1** | API Layer              | `app/api/`                         |
|   **Dev 2** | Retrieval (RAG)        | `app/rag/`                         |
|   **Dev 3** | LLM Integration        | `app/llm/`                         |
|   **Dev 4** | Data Processing        | `app/data/`                        |
|   **Dev 5** | Telegram Bot           | `app/bot/`                         |
|   **Dev 6** | Infrastructure & QA    | `app/core/`, `app/models/`, `tests/` |

Файлы, которые уже есть (не дублировать):

- app/core/ — конфигурация, логирование, сессии, исключения
- app/models/ — все Pydantic-модели
- app/metrics/ — сбор метрик
- app/main.py — точка входа FastAPI

Корень: .env.example, .gitignore, requirements.txt, pyproject.toml, README.md, CONTRIBUTING.md
CI/CD: .github/workflows/ci.yml (пока базовый вариант, позже доделать)

6. Правила взаимодействия компонентов
- Компоненты не обращаются к внутренним структурам друг друга
- Взаимодействие только через модели из app/models/
- RAG не знает о Telegram
- Telegram не знает о внутренней логике RAG
- LLM-провайдеры взаимозаменяемы (общий интерфейс)
- Все настройки из .env, все секреты через SecretStr
- Все ошибки логировать через get_logger(__name__)

7. Workflow разработки
- main — стабильная ветка 
- Feature-ветки: feature/api-layer, feature/rag-retriever и т.д.
- Pull Request в main → CI (ruff + pytest) → Code Review → Мёрж
- Перед коммитом: ruff check app/ --fix && pytest tests/

8. Язык
Системный промпт и ответы — только на русском
Модель эмбеддингов: intfloat/multilingual-e5-large (поддерживает русский)

---

## Документ 2: Интерфейсы для каждого разработчика — `INTERFACES.md`

```markdown
# Интерфейсы компонентов

## Dev 1 — API Layer (`app/api/`)

### Что использовать:
- `ChatRequest`, `ChatResponse`, `Source` из `app/models/chat.py`
- `ErrorResponse` из `app/models/error.py`
- `settings` из `app/core/config.py`
- `get_logger` из `app/core/__init__.py`

### Что нужно сделать:
- Endpoint `POST /chat`
- Валидация входа через `ChatRequest`
- Оркестрация: API → RAG → LLM → ответ
- Обработка ошибок через `AppException` и `ErrorResponse`
- Middleware для логирования запросов

### Что не нужно делать:
- Свои модели запросов/ответов (использовать из `app/models/`)
- Прямой импорт из `app/rag/` или `app/llm/` (использовать абстракции)

---

## Dev 2 — Retrieval / RAG (`app/rag/`)

### Что использовать:
- `RetrievedChunk` из `app/models/knowledge.py`
- `settings.retrieval_top_k`, `settings.chroma_persist_directory`, `settings.chroma_collection_name`
- `get_logger` из `app/core/__init__.py`

### Что нужно сделать:
- Функция `search_context(question: str, top_k: int = 5) -> list[RetrievedChunk]`
- Интеграция с ChromaDB
- Эмбеддинги через `sentence-transformers`
- Возвращать чанки с метаданными (включая `source_path`)

### Что не нужно делать:
- Знать о Telegram, API, боте
- Менять структуру метаданных (она приходит из Dev 4)

---

## Dev 3 — LLM Integration (`app/llm/`)

### Что использовать:
- `ChatRequest`, `ChatResponse`, `Source` из `app/models/chat.py`
- `RetrievedChunk` из `app/models/knowledge.py`
- `Session`, `DialogMessage` из `app/models/session.py`
- `settings.active_llm_provider`, `settings.fallback_answer`, `settings.max_history_length`
- `get_logger` из `app/core/__init__.py`
- openAI sdk и получается суть абстракции сводится к указанию base url + токена в разных провайдерах (коммент Николая)

### Что нужно сделать:
- `BaseLLMProvider` (абстрактный класс)
- `OpenAICompatibleProvider` (универсальный)
- `YandexGPTProvider`, `GigaChatProvider` (наследники)
- `build_prompt(question, context_chunks, history) -> str`
- Возвращать `fallback_answer` при низкой уверенности

### Что не нужно делать:
- Ходить в ChromaDB — получать чанки от Dev 2
- Знать о Telegram

---

## Dev 4 — Data Processing (`app/data/`)

### Что использовать:
- `KnowledgeDocument` из `app/models/knowledge.py`
- `settings.knowledge_base_file`, `settings.chunk_size`, `settings.chunk_overlap`
- `get_logger` из `app/core/__init__.py`

### Что нужно сделать:
- `build_knowledge_index(file_path: str) -> None`
- Загрузка txt из `knowledge_base/`
- Чанкинг с сохранением `source_path` и `category` в метаданных
- Индексация в ChromaDB

### Что не нужно делать:
- Менять формат метаданных (договориться с Dev 2)

---

## Dev 5 — Telegram Bot (`app/bot/`)

### Что использовать:
- `ChatRequest`, `ChatResponse`, `Source` из `app/models/chat.py`
- `settings` из `app/core/config.py`
- `get_logger` из `app/core/__init__.py`

### Что нужно сделать:
- Обработчики сообщений Telegram
- Отправка `ChatRequest` в Backend API (через httpx)
- Получение и хранение `session_id`
- Команды `/start`, обработка текста

### Что не нужно делать:
- Лесть в RAG или LLM напрямую
- Хранить историю диалогов локально (использовать `session_id`)

---

## Dev 6 — Infrastructure & QA (`app/core/`, `app/models/`, `tests/`)
**Уже готово.** Поддержка, тесты, документация.
```

