# Интеграция Retrieval (Dev 2) с векторным хранилищем

Документ для **Dev 2 (Retrieval / RAG)**: что уже сделано **Dev 4 (Data)**, какие контракты и конфиги использовать, и **что нужно реализовать** в коде поиска по Chroma.

---

## Чеклист Dev 2

1. Читать Chroma из **`settings.chroma_persist_directory`** и **`settings.chroma_collection_name`** (см. ниже).
2. Загружать **`SentenceTransformer(settings.embedding_model_name)`** — имя модели **должно совпадать** с индексацией.
3. Кодировать пользовательский запрос с префиксом **`query: `** (пробел после двоеточия — как у E5), **`normalize_embeddings=True`**.
4. Вызывать `collection.query(...)`, забирать **`n_results=settings.retrieval_top_k`**, в `include` как минимум `documents`, `metadatas`, `distances` (и при необходимости `ids`).
5. Собрать ответ слоя retrieval как список **`RetrievedChunk`** из `app/models/knowledge.py` (маппинг полей и **`score`** в диапазоне 0..1 — см. раздел про score).
6. Не менять схему metadata в Chroma без согласования (контракт с Dev 4 и тестами).

---

## Зона ответственности

| Команда | Зона |
|---------|------|
| **Dev 4** | Загрузка KB (YAML + `.txt`), чанкинг, эмбеддинги с `passage:`, инкрементальная запись в Chroma (`python -m app.data.indexing.pipeline`). |
| **Dev 2** | Подключение к той же коллекции, эмбед запроса с `query:`, similarity search, маппинг результатов в **`RetrievedChunk`**, передача в LLM / API. |

Индекс **не коммитится**: каталог `chroma_storage/` в `.gitignore`. Локально после `git pull` при отсутствии индекса — один прогон пайплайна индексации (или артефакт CI).

---

## Запуск через Docker Compose (Dev 4)

Для Dev 2 не обязательно настраивать venv с тяжёлыми зависимостями (torch, sentence-transformers) вручную — достаточно Docker.

### Предварительно

- Установлен и запущен **Docker Desktop** (или Colima / OrbStack).
- Склонирован репозиторий, текущая директория — корень проекта.

### Сборка образа (один раз или после изменения зависимостей)

```bash
docker compose -f compose.dev4.yaml build
```

### Индексация (создание / обновление `chroma_storage/`)

```bash
docker compose -f compose.dev4.yaml run --rm indexer
```

После успешного выполнения каталог `chroma_storage/` на хосте содержит готовый индекс — можно подключаться из кода Dev 2 через `chromadb.PersistentClient(path="chroma_storage")`.

Повторный запуск выполняет инкрементальное обновление (только изменённые документы).

### Тесты слоя данных

```bash
docker compose -f compose.dev4.yaml run --rm test-data
```

### Переменные окружения

По умолчанию пайплайн берёт значения из `Settings` (см. раздел ниже). Если нужно переопределить — передайте `.env`:

```bash
docker compose -f compose.dev4.yaml run --rm --env-file .env indexer
```

### Тома

| Том | Назначение |
|-----|------------|
| `./knowledge_base` (bind, ro) | Исходные документы и YAML-манифесты |
| `./chroma_storage` (bind, rw) | Персистентный индекс Chroma |
| `dev4_model_cache` (named) | Кэш моделей HuggingFace / torch (не скачивается повторно) |

---

## Карта кода (куда смотреть)

| Файл | Назначение |
|------|------------|
| `app/core/config.py` | **`Settings`**: пути Chroma, имя модели, `retrieval_top_k`, `knowledge_base_file` и др. |
| `app/models/knowledge.py` | **`RetrievedChunk`** — выход retrieval для LLM/API; **`IndexedChunk`** — формат чанка при индексации (для согласования полей с Chroma). |
| `app/data/indexing/pipeline.py` | Индексация: `add_e5_passage_prefix`, `run_indexing_pipeline`, инкремент по `doc_hash`. |
| `app/data/indexing/chroma_store.py` | **`add_indexed_chunks`** — как в Chroma попадают `ids` / `documents` / `metadatas` из **`IndexedChunk`**. |
| `app/data/indexing/chunking.py` | Только нарезка текста (внутренний тип `Chunk`); на retrieval не влияет. |
| `tests/data/test_incremental_chroma.py` | Поведение индекса (пустая коллекция / без изменений / смена файла). |

Конфиг в коде: **`from app.core.config import settings`** (singleton).

---

## Переменные окружения и поля `Settings`

Соответствие имени в `.env` и атрибута `settings` (см. также `.env.example`):

| Переменная `.env` | `settings.<атрибут>` | Назначение для Dev 2 |
|-------------------|----------------------|----------------------|
| `CHROMA_PERSIST_DIRECTORY` | `chroma_persist_directory` | Каталог персистентного клиента Chroma |
| `CHROMA_COLLECTION_NAME` | `chroma_collection_name` | Имя коллекции |
| `EMBEDDING_MODEL_NAME` | `embedding_model_name` | Модель эмбеддингов (**та же**, что при индексации) |
| `RETRIEVAL_TOP_K` | `retrieval_top_k` | Сколько чанков запрашивать у `query` |
| `KNOWLEDGE_BASE_FILE` | `knowledge_base_file` | Только для пайплайна индексации; retrieval **не читает** KB с диска |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `chunk_size` / `chunk_overlap` | Только **индексация**; на формат векторов в Chroma retrieval не влияет |

Подключение к коллекции (чтение):

```python
import chromadb
from app.core.config import settings

client = chromadb.PersistentClient(path=settings.chroma_persist_directory)
collection = client.get_collection(name=settings.chroma_collection_name)
```

Если коллекции ещё нет, `get_collection` упадёт — это ожидаемо до первой успешной индексации.

---

## Схема записи в Chroma (контракт с индексатором)

Коллекция создаётся с **`metadata={"hnsw:space": "cosine"}`**; эмбеддинги при индексации **нормализованы**.

У каждой записи (один чанк):

| Chroma | Источник смысла |
|--------|-----------------|
| **id** | `{source_path}:{chunk_index}` — совпадает с **`IndexedChunk.chunk_id`** |
| **document** | Текст чанка **без** префикса `passage:` — совпадает с **`IndexedChunk.text`** |
| **embedding** | Вектор из модели по строке `passage: {text}` |
| **metadata** | Словарь из **`IndexedChunk.metadata`** (типы значений: `str`, `int`, `float`, `bool` только; **`null` нет**) |

Обязательные / частые ключи **metadata**:

- `source_path` — абсолютный путь к исходному `.txt` (как в репозитории после резолва).
- `chunk_index` — номер чанка внутри документа (дублирует суффикс в `id`, удобно для отладки).
- `doc_hash` — SHA-256 текста **целого** документа (инкремент у Dev 4).
- `doc_id`, `doc_type`, `location`, `yaml_path` — **только если** были в манифесте YAML; для точки входа одним `.txt` этих ключей может не быть.

---

## Критично: префиксы E5

Индексация считает вектор по строке **`passage: `** + текст (один пробел после двоеточия — функция `add_e5_passage_prefix` в `pipeline.py`).

Поиск **обязан** использовать ту же модель и префикс **`query: `** + текст пользователя:

```python
from sentence_transformers import SentenceTransformer
from app.core.config import settings

model = SentenceTransformer(settings.embedding_model_name)
q = model.encode(
    [f"query: {user_text}"],
    normalize_embeddings=True,
    show_progress_bar=False,
    convert_to_numpy=True,
)[0].tolist()
```

Дальше (пример вызова Chroma):

```python
raw = collection.query(
    query_embeddings=[q],
    n_results=settings.retrieval_top_k,
    include=["ids", "documents", "metadatas", "distances"],
)
```

Без пары **`passage:`** (индекс) / **`query:`** (поиск) качество retrieval резко падает.

---

## Маппинг результата Chroma в `RetrievedChunk`

Модель: **`app/models/knowledge.py` → `RetrievedChunk`**: `chunk_id`, `text`, `metadata`, **`score`** ∈ [0, 1].

Рекомендуемый маппинг полей:

- `chunk_id` ← элемент из `raw["ids"][i]` (первая выдача — список списков по запросам).
- `text` ← соответствующий `documents`.
- `metadata` ← словарь `metadatas` **как есть** (для ссылок на источник обычно достаточно `source_path`).

**`score`:** в `RetrievedChunk` задокументирован диапазон 0..1. Chroma отдаёт **`distances`** в метрике коллекции (косинусное расстояние / связанная величина — зависит от версии клиента). Нужно **в одном месте** в коде retrieval зафиксировать формулу перевода `distance → score` (например, монотонное преобразование в 0..1) и не смешивать разные трактовки в разных endpoint’ах. До фиксации формулы — согласовать с командой и кратко продублировать формулу здесь в PR.

---

## Инкрементальная индексация (что важно для Dev 2)

Повторный запуск `python -m app.data.indexing.pipeline`:

- удаляет чанки по `source_path`, которых больше нет в текущем наборе документов;
- пересчитывает эмбеддинги только для документов с **изменённым** `doc_hash`;
- при отсутствии изменений в логах — сообщение о пропуске записи.

Dev 2 **только читает** коллекцию; актуальность индекса обеспечивает Dev 4 / CI.

Ограничение: если меняются только **`CHUNK_SIZE` / `CHUNK_OVERLAP`**, текст файлов и `doc_hash` не меняются — инкремент может **не** пересобрать чанки. Тогда нужна полная переиндексация или отдельный сценарий (например, флаг force reindex — на будущее).

---

## Проверки (Dev 2 и общие)

Убедиться, что индекс не пустой (подставьте свои значения из `.env` при отличии от умолчанию):

```bash
python -c "
import chromadb
from app.core.config import settings
c = chromadb.PersistentClient(path=settings.chroma_persist_directory)
col = c.get_collection(settings.chroma_collection_name)
print('count:', col.count())
"
```

Чанки не шире окна модели по токенам (`CHUNK_SIZE` в символах, лимит E5 в токенах):

```bash
pytest tests/data/test_embedding_chunk_budget.py -v
```

Эти тесты помечены **`slow`**. Быстрый CI-набор без них:

```bash
ruff check app/ tests/
pytest tests/ -m "not slow" -v
```

Полный прогон всех тестов:

```bash
pytest tests/ -v
```

**Через Docker (без локального venv):**

```bash
docker compose -f compose.dev4.yaml run --rm test-data
```

---

## Смена контракта

Любые изменения полей metadata, формата `id` или префиксов E5 — **согласовать с Dev 4**, обновить тесты в `tests/data/` и этот документ; модели — в `app/models/`, как в CONTRIBUTING.
