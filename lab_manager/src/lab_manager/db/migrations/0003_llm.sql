-- Кеш результатов LLM (ключ: хеш входа + версия промпта + модель) и учёт каждого вызова.

CREATE TABLE llm_cache (
    cache_key  TEXT PRIMARY KEY,
    purpose    TEXT NOT NULL,
    value      TEXT NOT NULL,  -- JSON
    created_at TEXT NOT NULL
);

CREATE TABLE llm_calls (
    id                INTEGER PRIMARY KEY,
    purpose           TEXT NOT NULL,
    model             TEXT NOT NULL,
    prompt_version    TEXT NOT NULL,
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    latency_ms        INTEGER NOT NULL,
    ok                INTEGER NOT NULL CHECK (ok IN (0, 1)),
    error             TEXT,
    created_at        TEXT NOT NULL
);
CREATE INDEX llm_calls_created ON llm_calls (created_at);
