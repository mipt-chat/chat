# Интеграция Retrieval (Dev 2) с векторным хранилищем

Документ описывает, что подготовил **Data Processing (Dev 4)**, и как **Retrieval (Dev 2)** подключается к **ChromaDB** и эмбеддингам.

## Что уже есть в репозитории

- Код индексации: `app/data/indexing/pipeline.py`
- Чанкинг: `app/data/indexing/chunking.py`
- Запуск полной / инкрементальной индексации:

  ```bash
  python -m app.data.indexing.pipeline
  ```

- База знаний: `knowledge_base/root.yaml` и связанные YAML + `.txt` (см. README).

Индекс **не коммитится** в git: каталог `chroma_storage/` в `.gitignore`. Каждый разработчик или CI после `git pull` при необходимости **один раз** (или после изменений KB) запускает команду выше.

## Переменные окружения (те же, что у индексации)

Все значения читаются из `app/core/config.py` (`Settings`) и из `.env` (пример — `.env.example`).

| Переменная | Назначение |
|------------|------------|
| `CHROMA_PERSIST_DIRECTORY` | Каталог персистентного хранилища Chroma (по умолчанию `chroma_storage`) |
| `CHROMA_COLLECTION_NAME` | Имя коллекции (по умолчанию `support_knowledge`) |
| `EMBEDDING_MODEL_NAME` | Модель эмбеддингов, **та же**, что при индексации (по умолчанию `intfloat/multilingual-e5-base`) |
| `RETRIEVAL_TOP_K` | Сколько чанков забирать в контекст (по умолчанию `5`) |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Используются только при **индексации**; на формат векторов в Chroma не влияют |

## Проверка: чанки не шире окна эмбеддинг-модели

`CHUNK_SIZE` задан в **символах**, лимит E5 — в **токенах** (`max_seq_length`, обычно 512). Чтобы убедиться, что при индексации строки `passage: …` не обрезаются токенизатором:

```bash
pytest tests/data/test_embedding_chunk_budget.py -v
```

Тесты помечены `slow` (загружается `SentenceTransformer`). Быстрый прогон без них: `pytest tests/ -m "not slow"`.

Подключение к Chroma в коде (пример):

```python
import chromadb

client = chromadb.PersistentClient(path="<CHROMA_PERSIST_DIRECTORY>")
collection = client.get_collection(name="<CHROMA_COLLECTION_NAME>")
```

## Критично: префиксы E5

При индексации каждый чанк эмбеддится как строка с префиксом **`passage:`** (см. `add_e5_passage_prefix` в `pipeline.py`). В поле `documents` в Chroma сохраняется **текст чанка без префикса**; префикс участвует только в расчёте вектора.

Для **запроса пользователя** при поиске нужно использовать ту же модель и префикс **`query:`**:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(settings.embedding_model_name)
q = model.encode(
    [f"query: {user_text}"],
    normalize_embeddings=True,
    convert_to_numpy=True,
)[0].tolist()
```

Дальше — `collection.query(query_embeddings=[q], n_results=settings.retrieval_top_k, include=["documents", "metadatas", "distances"])` (или эквивалентный API).

Без согласованности **`passage:`** (индекс) / **`query:`** (поиск) качество retrieval заметно падает.

## Содержимое коллекции

У каждой записи (чанка):

- **id**: строка вида `{source_path}:{chunk_index}`
- **embedding**: вектор чанка (нормализованный, косинусная метрика в метаданных коллекции: `hnsw:space` = `cosine`)
- **document**: текст чанка (без префикса `passage:`)
- **metadata** (словарь), ключевые поля:
  - `source_path` — путь к исходному `.txt`
  - `chunk_index`, `start_char`, `end_char`
  - `doc_id` — идентификатор из YAML (`docs`), если документ из манифеста
  - `doc_type`, `location`, `yaml_path` — при наличии в манифесте
  - `doc_hash` — SHA-256 содержимого **целого документа** (для инкрементальной индексации)

## Модель ответа API (контракт между слоями)

Структура для передачи в LLM / API описана в `app/models/knowledge.py` — класс **`RetrievedChunk`**: `chunk_id`, `text`, `metadata`, `score`.

При маппинге из Chroma:

- `chunk_id` ← id записи в Chroma;
- `text` ← `document`;
- `metadata` ← метаданные чанка (обязательно сохранять `source_path` для ссылок на источник);
- `score` — привести расстояние Chroma к ожидаемому диапазону `RetrievedChunk` (в модели указан диапазон 0..1; уточните с командой, если используете «сырое» расстояние вместо similarity).

## Инкрементальная индексация (для понимания поведения)

Повторный запуск `python -m app.data.indexing.pipeline`:

- удаляет векторы документов, которых больше нет в манифесте;
- пересчитывает только документы с изменённым текстом (по `doc_hash`);
- при отсутствии изменений может завершиться без записи в Chroma.

Dev 2 **только читает** коллекцию; пересборку индекса инициирует Dev 4 / CI / админ.

## Проверка, что индекс на месте

```bash
python -c "
import chromadb
c = chromadb.PersistentClient(path='chroma_storage')
col = c.get_collection('support_knowledge')
print('count:', col.count())
"
```

Подставьте свои `CHROMA_PERSIST_DIRECTORY` и `CHROMA_COLLECTION_NAME`, если отличаются от значений по умолчанию.

## Ограничение (на будущее)

Если меняются только **`CHUNK_SIZE` / `CHUNK_OVERLAP`**, хэш файла не меняется — инкремент может **не** пересобрать чанки. Тогда нужна полная переиндексация (очистка коллекции или отдельный сценарий). С Dev 4 можно согласовать отдельный флаг «force reindex», если понадобится.

---

Вопросы по формату метаданных или изменению контракта — через команду и правки в `app/models/`, как в CONTRIBUTING.
