# LangChain Search Agent

Этот проект представляет собой небольшой агент на базе LangChain, который использует Groq-модель и DuckDuckGo для поиска информации в интернете и последующего суммаризирования содержимого страницы.

## Что делает скрипт

Файл [agent_search.py](agent_search.py) выполняет следующие шаги:

- загружает переменные окружения из `.env`;
- создает модель Groq через `ChatGroq`;
- использует поисковый инструмент DuckDuckGo;
- загружает найденную страницу и преобразует HTML в текст;
- запускает агент с системным промптом, который просит искать информацию на языке пользователя;
- выполняет один конкретный запрос: поиск официального сайта LangChain, чтение текста внутри страницы и краткое резюме.

Сейчас это демонстрационный сценарий, а не универсальный чат-бот для любых запросов.

## Технологии

- Python 3
- LangChain
- Groq
- DuckDuckGo Search
- Async HTML Loader
- Html2TextTransformer

## Установка

1. Установите зависимости:

```bash
pip install -r requirements.txt
```

2. Создайте файл `.env` на основе шаблона:

```bash
cp .env.example .env
```

3. Заполните переменные окружения:

```env
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT="Your Project Name"

GROQ_API_KEY=your_groq_api_key_here
```

## Запуск

```bash
python3 agent_search.py
```

При запуске скрипт автоматически выполняет запрос:

```text
Найди официальный вебсайт langchain, прочитай текст внутри и сделай саммари.
```

## Как работает агент

Внутри `agent_search.py` используется:

- `load_dotenv()` для чтения `.env`;
- `ChatGroq(api_key=..., model="qwen/qwen3.8-27b", temperature=0.6, max_tokens=1000, streaming=True, verbose=True)`;
- `DuckDuckGoSearchAPIWrapper(region="rus", max_results=5)`;
- `DuckDuckGoSearchResults` как инструмент поиска;
- `AsyncHtmlLoader` и `Html2TextTransformer` для очистки HTML и получения текста страницы.

Агент создается так:

```python
agent = create_agent(
    model,
    tools=[duckduckgo_tool, html2text_tool],
    system_prompt=system_prompt,
)
```

## Примечание

Этот проект хорошо подходит как базовый пример сочетания:

- Groq-модели для генерации ответа;
- LangChain-агента для работы с инструментами;
- внешнего поиска через DuckDuckGo;
- чтения и анализа HTML-содержимого страницы.

Если нужно, его можно расширить до более общего интерфейса с пользовательским вводом вместо жестко заданного запроса.
