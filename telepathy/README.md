# telepathy

Telegram-бот «Телепатия» для пары. Двое независимо пишут боту в личку случайные мысли и не
видят чужих. Если мысли партнёров совпали по смыслу, бот сообщает обоим и показывает обе мысли.
Мысли без совпадения партнёру не раскрываются никогда.

## Как это работает

1. `/start` регистрирует, `/pair` выдаёт код-приглашение, партнёр отправляет `/join КОД`.
   В паре строго два человека: в свою пару вступить нельзя, третьего не пустит.
2. Любой текст — мысль. Бот сохраняет текст, эмбеддинг (`intfloat/multilingual-e5-small`,
   квантованная ONNX-версия, локально) и время и отвечает «🔮 мысль принята».
   Не больше `MAX_THOUGHTS_PER_HOUR` (30) мыслей от одного человека за скользящий час: лишняя
   не сохраняется, бот отвечает, через сколько минут можно снова.
3. Сразу после сохранения ищется совпадение. Кандидаты — мысли партнёра за последние
   `MATCH_WINDOW_HOURS` часов, ещё ни с чем не совпавшие. Для каждой
   `score = cos(a, b) · exp(−Δt / TAU_HOURS)`; косинус считает sqlite-vec.
4. Если лучший `score ≥ SCORE_THRESHOLD`, пара мыслей уходит LLM-судье (PydanticAI,
   структурированный вывод `Match`: `is_same`, `common_idea`, `message`). Судья учитывает
   отрицания: «хочу пиццу» и «не хочу пиццу» — разные мысли.
5. При `is_same=true` обе мысли помечаются совпавшими, а оба партнёра получают игривое
   сообщение судьи и обе мысли.
6. `/stats`: число совпадений, процент совпавших мыслей за 7 дней, самая быстрая синхронность.

### Отладка

Каждому участнику пары доступны команды с цифрами (новые записи сверху, `N` — сколько, до 30):

- `/thoughts [N]` — последние мысли пары со временем;
- `/debug [N]` — журнал поиска совпадения по каждой мысли: лучший кандидат, cos, Δt, score,
  порог, число кандидатов и решение (нет кандидатов / ниже порога / судья: разные /
  судья недоступен / совпало);
- `/matches [N]` — совпадения с cos, score и Δt.

Правило приватности то же, что у бота: текст мысли виден, только если она своя или уже совпала.
Несовпавшие мысли партнёра показываются как «🔒 скрыто» — со временем и цифрами, без текста.
Журнал хранится в таблице `match_checks`.

Если LLM недоступна, мысль всё равно сохраняется, бот не падает, ошибка пишется в лог.
Тексты мыслей в логи не попадают.

## Запуск (Docker)

```bash
cd ~/projects/telepathy
cp .env.example .env && chmod 600 .env   # токен бота, ALLOWED_USER_IDS, ключ LLM
docker compose up -d --build             # сборка и запуск
docker compose ps                        # статус и healthcheck
docker compose logs -f telepathy         # логи
docker compose down                      # остановить (данные остаются в томе)
```

- Первый запуск скачивает модель эмбеддингов (~120 МБ: int8 ONNX и SentencePiece) с Hugging Face
  в том `telepathy_telepathy_data` (`/data/models`); дальше она грузится из тома без сети.
- Данные (SQLite) — в том же томе, `/data/telepathy.db`.
- Секреты — только через `env_file: .env`, в образ не попадают (`.dockerignore`).
- Сертификат Минцифры для GigaChat монтируется только для чтения; путь на хосте —
  `GIGACHAT_CA_HOST_FILE` (по умолчанию `/home/agent/certs/Russian_Trusted_Root_CA.cer`).
- Лимиты: 600 МБ памяти, 0.75 CPU; корневая ФС только для чтения, `cap_drop: ALL`,
  `no-new-privileges`; `restart: unless-stopped`; healthcheck — `python -m telepathy.health`
  (свежесть файла-пульса, без загрузки модели).
- Портов наружу нет.

### Два режима

| `TELEGRAM_MODE` | Как получает апдейты | Когда |
|---|---|---|
| `polling` (по умолчанию, так и развёрнуто) | сам опрашивает Telegram (long polling) | отдельный бот со своим токеном — сейчас @cherryegoricbot |
| `gateway` | шлюз [lab_manager](../lab_manager/) шлёт апдейты POST-ом на `telepathy:9010/telegram/update` | если «Телепатию» захочется вернуть в общий бот |

В режиме `gateway` токен тот же, что у общего бота: опрашивает Telegram только шлюз, а отвечает
«Телепатия» сама через Bot API; шлюз и «Телепатия» сверяют общий секрет (`GATEWAY_SECRET` здесь =
`TELEPATHY_SECRET` в `lab_manager/.env`). Для этого режима нужна общая docker-сеть с lab_manager.

### LLM-судья: DeepSeek и GigaChat

- `LLM_MODEL=deepseek:deepseek-chat` + `LLM_API_KEY` — основной судья DeepSeek; если задан ещё
  и `GIGACHAT_CREDENTIALS`, GigaChat — запасной: отвечает при любой ошибке DeepSeek.
- `LLM_MODEL=gigachat:GigaChat-2` — только GigaChat.
- Нет ключа DeepSeek, но есть GigaChat — судит GigaChat. Не настроено ничего — судья
  отключён (ошибка в логе), мысли принимаются, совпадений нет.

GigaChat использует свой протокол `functions`, поэтому подключён через `FunctionModel`-адаптер
(`llm/`, перенесён из lab_manager вместе с тестами на фейковом сервере).

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
| `GIGACHAT_CREDENTIALS` | — | ключ авторизации GigaChat (основной или запасной судья) |
| `GIGACHAT_SCOPE` / `GIGACHAT_MODEL` | `GIGACHAT_API_PERS` / `GigaChat-2` | scope и модель GigaChat |
| `GIGACHAT_CA_BUNDLE_FILE` | — | сертификат Минцифры (PEM), в Docker монтируется сам |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-small` | репозиторий модели на Hugging Face |
| `EMBEDDING_ONNX_FILE` | `onnx/model_qint8_avx512_vnni.onnx` | файл ONNX-модели в репозитории |
| `EMBEDDING_THREADS` | `1` | потоков ONNX Runtime |
| `MATCH_WINDOW_HOURS` | `24` | окно поиска кандидатов |
| `TAU_HOURS` | `24` | τ затухания по времени |
| `SCORE_THRESHOLD` | `0.75` | порог score для обращения к судье |
| `MAX_THOUGHTS_PER_HOUR` | `30` | лимит мыслей одного человека за скользящий час |
| `TZ` | `Europe/Moscow` | часовой пояс |
| `DATA_DIR` | `data` | SQLite и кеш модели |
| `LOG_LEVEL` | `INFO` | уровень логов |

## Калибровка и ресурсы

VPS маленький (2 ГБ RAM, 1 CPU), поэтому эмбеддинги считает ONNX Runtime без torch и
sentence-transformers: квантованная (int8) e5-small из того же репозитория, токенизатор —
SentencePiece (id токенов совпадают с XLM-R), пулинг и нормализация — как в sentence-transformers.

Замеры на этом VPS (процесс целиком, как при старте):

| Вариант | RAM процесса | Эмбеддинг мысли |
|---|---|---|
| torch + sentence-transformers, fp32 (было) | ~850 МБ | ~70 мс |
| ONNX Runtime, int8 + SentencePiece (сейчас) | ~450 МБ, пик ~480 | ~20 мс |

Из 450 МБ: импорт aiogram, PydanticAI и SQLAlchemy ~220 (из них aiogram ~150), SentencePiece ~60,
ONNX-сессия ~140. Токенизатор `tokenizers` от Hugging Face на этом словаре занимал бы ~280 МБ.

Косинусы int8-модели отличаются от fp32 не больше чем на 0.01. Они сжаты: даже несвязанные
фразы дают ≈ 0.82, перефразы — 0.86–0.95, а «хочу пиццу» / «не хочу пиццу» — 0.95. Поэтому порог
0.75 отсекает в основном по времени (при τ = 24 ч несвязанная мысль проходит порог, если
Δt ≲ 2 ч), а смысл и отрицания решает судья. Чтобы реже звать LLM, поднимите `SCORE_THRESHOLD`
до ~0.85.

## Архитектура

```
src/telepathy/
  config.py          настройки (pydantic-settings)
  bot/               хендлеры aiogram, тексты, приём апдейтов от шлюза
  services/          pairing, matching, scoring, embeddings, judge, stats
  db/                модели SQLAlchemy 2 (async), sqlite-vec, репозитории
  llm/               клиент GigaChat и адаптер для PydanticAI
  health.py          healthcheck без тяжёлых импортов
```

Хендлеры тонкие, логика — в `services/`. Модель эмбеддингов грузится один раз при старте,
`encode` выполняется в `asyncio.to_thread`. Поиск совпадения для пары сериализован, поэтому
одновременные мысли обоих партнёров дают ровно одно совпадение.

## Разработка

```bash
uv sync                                     # зависимости (без torch)
uv run pytest                               # тесты без сети: TestModel/FunctionModel, фейковый Telegram
uv run pytest -m live                       # с настоящей моделью эмбеддингов (скачает её)
uv run ruff check . && uv run ruff format --check .
```
