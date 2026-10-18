# lab_manager

Telegram-менеджер лабораторных работ. Принимает задания, раскладывает их по папкам,
синхронизирует с Яндекс Диском, ведёт статусы и готовит промпты для агента-исполнителя.
Саму лабу не выполняет. Подробности — в [REQUIREMENTS.md](REQUIREMENTS.md).

> Статус: в разработке (MVP, шаг 1 — каркас и конфигурация).

## Требования

- Python 3.12+, [uv](https://docs.astral.sh/uv/)
- Docker (rootless) с `docker compose` — для запуска в контейнере

## Быстрый старт (разработка)

```bash
cd ~/projects/lab_manager
uv sync                      # зависимости + dev-инструменты
cp .env.example .env && chmod 600 .env   # заполнить секреты
uv run pytest                # тесты (внешние API замоканы)
uv run ruff check . && uv run ruff format --check .
```

## Переменные окружения

Полный список с комментариями — в [.env.example](.env.example).

| Переменная | Обязательна | По умолчанию | Назначение |
|---|---|---|---|
| `TELEGRAM_BOT_TOKEN` | да | — | токен бота |
| `TELEGRAM_OWNER_ID` | да | — | Telegram ID владельца (единственный пользователь) |
| `YANDEX_DISK_TOKEN` | да | — | OAuth-токен Яндекс Диска |
| `YANDEX_DISK_ROOT` | нет | `disk:/Лабы` | корневая папка на Диске |
| `GIGACHAT_CREDENTIALS` | да | — | ключ авторизации GigaChat |
| `GIGACHAT_SCOPE` | нет | `GIGACHAT_API_PERS` | scope GigaChat |
| `GIGACHAT_MODEL` | нет | `GigaChat-2` | модель |
| `TZ` | нет | `Europe/Moscow` | часовой пояс для дедлайнов |
| `DATA_DIR` | нет | `data` | каталог для SQLite и файлов |
| `LOG_LEVEL` | нет | `INFO` | уровень логирования |

Секреты хранятся только в `.env` (`chmod 600`, в `.gitignore`) и в логи не выводятся.

## Порты

MVP не занимает портов (Telegram — polling). См. [PORTS.md](PORTS.md).

## Разработка

- TDD: сначала тест, потом код.
- `uv run pytest -m live` — тесты с реальными API (нужны ключи в `.env`), вручную.
