# Customer Support AI Bot

MVP-проект чат-бота клиентской поддержки с использованием RAG-подхода.

## 🚀 Быстрый старт

### Предварительные требования
- Python 3.10+
- pip
- Git

### Установка

1. Клонируйте репозиторий:
   ```commandline
   git clone <repository-url>
   cd project
   ```
   
2. Создайте и активируйте виртуальное окружение:
- Рекомендуется использовать uv:
```commandline
uv venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
```

- Возможен также классический подход: 
```commandline
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

3. Установите зависимости
- uv:
```commandline
uv pip install -r requirements.txt
```
-классика:
```commandline
pip install -r requirements.txt
```
4. Настройте переменные окружения:
```commandline
cp .env.example .env
# Отредактируйте .env, указав ключи API
```

5. Поместите базу знаний:
```commandline
# Скопируйте ваш knowledge.txt в папку knowledge_base/
cp /path/to/knowledge.txt knowledge_base/
```

6. Запустите индексацию базы знаний:
```commandline
python -m app.data.indexing.pipeline
```

7. Запустите приложение:
```commandline
python -m app.main
```

## Структура проекта
Подробное описание структуры и правил разработки см. в CONTRIBUTING.md.

## Технологический стек

- Backend: Python 3.10+, FastAPI, Uvicorn
- LLM: YandexGPT, GigaChat (OpenAI-совместимый интерфейс)
- Vector DB: ChromaDB
- Embeddings: sentence-transformers
- Telegram Bot: aiogram
- Testing: pytest