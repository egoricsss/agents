# telepathy

Telegram-бот «Телепатия» для пары. Двое независимо пишут боту в личку случайные мысли и не
видят чужих. Если мысли партнёров совпали по смыслу, бот сообщает обоим и показывает обе мысли.
Мысли без совпадения партнёру не раскрываются никогда.

## Как это работает

1. `/start` регистрирует, `/pair` выдаёт код-приглашение, партнёр отправляет `/join КОД`.
   В паре строго два человека: в свою пару вступить нельзя, третьего не пустит.
2. Любой текст — мысль. Бот сохраняет текст, эмбеддинг (`intfloat/multilingual-e5-small`,
   локально) и время и отвечает «🔮 мысль принята».
3. Сразу после сохранения ищется совпадение. Кандидаты — мысли партнёра за последние
   `MATCH_WINDOW_HOURS` часов, ещё ни с чем не совпавшие. Для каждой
   `score = cos(a, b) · exp(−Δt / TAU_HOURS)`; косинус считает sqlite-vec.
4. Если лучший `score ≥ SCORE_THRESHOLD`, пара мыслей уходит LLM-судье (PydanticAI,
   структурированный вывод `Match`: `is_same`, `common_idea`, `message`). Судья учитывает
   отрицания: «хочу пиццу» и «не хочу пиццу» — разные мысли.
5. При `is_same=true` обе мысли помечаются совпавшими, а оба партнёра получают игривое
   сообщение судьи и обе мысли.
6. `/stats`: число совпадений, процент совпавших мыслей за 7 дней, самая быстрая синхронность.

Если LLM недоступна, мысль всё равно сохраняется, бот не падает, ошибка пишется в лог.
Тексты мыслей в логи не попадают.

## Запуск (Docker)

```bash
cd ~/projects/telepathy
cp .env.example .env && chmod 600 .env   # заполнить TELEGRAM_BOT_TOKEN и LLM_API_KEY
docker compose up -d --build             # сборка и запуск
docker compose ps                        # статус и healthcheck
docker compose logs -f telepathy         # логи
docker compose down                      # остановить (данные остаются в томе)
```

- Первый запуск скачивает модель эмбеддингов (~470 МБ) с Hugging Face в том
  `telepathy_telepathy_data` (`/data/models`); дальше она грузится из тома без сети.
- Данные (SQLite) — в том же томе, `/data/telepathy.db`.
- Секреты — только через `env_file: .env`, в образ не попадают (`.dockerignore`).
- Лимиты: 1 ГБ памяти, 0.75 CPU; корневая ФС только для чтения, `cap_drop: ALL`,
  `no-new-privileges`; `restart: unless-stopped`; healthcheck — `python -m telepathy.health`
  (свежесть файла-пульса, без импорта torch).
- Портов наружу нет.

### Два режима

| `TELEGRAM_MODE` | Как получает апдейты | Когда |
|---|---|---|
| `polling` (по умолчанию) | сам опрашивает Telegram (long polling) | отдельный бот со своим токеном |
| `gateway` | шлюз [lab_manager](../lab_manager/) шлёт апдейты POST-ом на `telepathy:9010/telegram/update` в docker-сети `agents` | общий бот владельца |

В режиме `gateway` токен тот же, что у общего бота: опрашивает Telegram только шлюз, а отвечает
«Телепатия» сама через Bot API. Шлюз и «Телепатия» проверяют общий секрет
(`GATEWAY_SECRET` здесь = `TELEPATHY_SECRET` в `lab_manager/.env`, заголовок
`X-Telegram-Bot-Api-Secret-Token`). Кто имеет доступ, решает шлюз: владельцу — всё,
гостям из `TELEPATHY_USERS` — только «Телепатия». Сеть `agents` создаёт этот compose-проект,
поэтому его надо поднять раньше lab_manager.

## Переменные окружения

Полный список с комментариями — в [.env.example](.env.example).

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | — | токен бота (обязателен) |
| `TELEGRAM_MODE` | `polling` | `polling` или `gateway` |
| `ALLOWED_USER_IDS` | пусто (все) | кому можно пользоваться ботом, через запятую |
| `GATEWAY_SECRET` | — | общий секрет со шлюзом (обязателен в `gateway`) |
| `GATEWAY_HOST` / `GATEWAY_PORT` | `0.0.0.0` / `9010` | где слушать шлюз внутри контейнера |
| `LLM_MODEL` | `deepseek:deepseek-chat` | модель судьи, строка PydanticAI «провайдер:модель» |
| `LLM_API_KEY` | — | ключ провайдера LLM |
| `LLM_BASE_URL` | — | адрес OpenAI-совместимого API (с `LLM_MODEL=openai-chat:<модель>`) |
| `LLM_TIMEOUT_SECONDS` | `30` | таймаут запроса к судье |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-small` | модель sentence-transformers |
| `EMBEDDING_THREADS` | `1` | потоков torch |
| `MATCH_WINDOW_HOURS` | `24` | окно поиска кандидатов |
| `TAU_HOURS` | `24` | τ затухания по времени |
| `SCORE_THRESHOLD` | `0.75` | порог score для обращения к судье |
| `TZ` | `Europe/Moscow` | часовой пояс |
| `DATA_DIR` | `data` | SQLite и кеш модели |
| `LOG_LEVEL` | `INFO` | уровень логов |

## Калибровка и ресурсы

Замеры на этом VPS (CPU, fp32):

- **Память:** torch + sentence-transformers + e5-small ≈ 0.8 ГБ RSS (импорт библиотек
  ~0.4 ГБ, модель ~0.35 ГБ). Отсюда `mem_limit: 1g`: при перерасходе OOM убьёт только этот
  контейнер. Эмбеддинг одной мысли ≈ 70 мс.
- **Косинусы e5 сжаты:** даже несвязанные фразы дают ≈ 0.82, перефразы — 0.86–0.95,
  а «хочу пиццу» / «не хочу пиццу» — 0.95. Поэтому порог 0.75 отсекает в основном по времени
  (при τ = 24 ч несвязанная мысль проходит порог, если Δt ≲ 2 ч), а смысл и отрицания решает
  судья. Чтобы реже звать LLM, поднимите `SCORE_THRESHOLD` до ~0.85.

## Архитектура

```
src/telepathy/
  config.py          настройки (pydantic-settings)
  bot/               хендлеры aiogram, тексты, приём апдейтов от шлюза
  services/          pairing, matching, scoring, embeddings, judge, stats
  db/                модели SQLAlchemy 2 (async), sqlite-vec, репозитории
  health.py          healthcheck без тяжёлых импортов
```

Хендлеры тонкие, логика — в `services/`. Модель эмбеддингов грузится один раз при старте,
`encode` выполняется в `asyncio.to_thread`. Поиск совпадения для пары сериализован, поэтому
одновременные мысли обоих партнёров дают ровно одно совпадение.

## Разработка

```bash
uv sync                                     # зависимости (torch — CPU-сборка)
uv run pytest                               # тесты без сети: TestModel/FunctionModel, фейковый Telegram
uv run pytest -m live                       # с настоящей моделью эмбеддингов (скачает её)
uv run ruff check . && uv run ruff format --check .
```
