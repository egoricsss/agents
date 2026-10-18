-- Начальная схема. Время хранится как ISO 8601 в UTC с микросекундами,
-- поэтому строковое сравнение совпадает с хронологическим.

CREATE TABLE subjects (
    id         INTEGER PRIMARY KEY,
    name       TEXT NOT NULL,
    name_key   TEXT NOT NULL UNIQUE,  -- нормализованное имя: casefold + схлопнутые пробелы
    created_at TEXT NOT NULL
);

CREATE TABLE labs (
    id         INTEGER PRIMARY KEY,
    subject_id INTEGER NOT NULL REFERENCES subjects (id) ON DELETE RESTRICT,
    number     INTEGER NOT NULL CHECK (number >= 1),
    title      TEXT NOT NULL,
    status     TEXT NOT NULL DEFAULT 'todo'
               CHECK (status IN ('todo', 'in_progress', 'blocked', 'done', 'submitted', 'defended')),
    deadline   TEXT,
    summary    TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (subject_id, number)
);
CREATE INDEX labs_deadline ON labs (deadline);

CREATE TABLE status_history (
    id         INTEGER PRIMARY KEY,
    lab_id     INTEGER NOT NULL REFERENCES labs (id) ON DELETE CASCADE,
    old_status TEXT,
    new_status TEXT NOT NULL,
    source     TEXT NOT NULL CHECK (source IN ('telegram', 'mcp', 'executor', 'system')),
    changed_at TEXT NOT NULL
);
CREATE INDEX status_history_lab ON status_history (lab_id, id);

CREATE TABLE files (
    id            INTEGER PRIMARY KEY,
    lab_id        INTEGER NOT NULL REFERENCES labs (id) ON DELETE CASCADE,
    sha256        TEXT NOT NULL UNIQUE CHECK (length(sha256) = 64),
    version       INTEGER NOT NULL CHECK (version >= 1),
    original_name TEXT NOT NULL,
    mime_type     TEXT,
    size_bytes    INTEGER NOT NULL CHECK (size_bytes >= 0),
    local_path    TEXT NOT NULL,
    disk_path     TEXT,
    uploaded_at   TEXT,
    created_at    TEXT NOT NULL,
    UNIQUE (lab_id, version)
);

CREATE TABLE requirements (
    id         INTEGER PRIMARY KEY,
    scope      TEXT NOT NULL CHECK (scope IN ('global', 'subject', 'lab')),
    subject_id INTEGER REFERENCES subjects (id) ON DELETE CASCADE,
    lab_id     INTEGER REFERENCES labs (id) ON DELETE CASCADE,
    target_key TEXT NOT NULL,  -- 'global' | 'subject:<id>' | 'lab:<id>'
    version    INTEGER NOT NULL CHECK (version >= 1),
    text       TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (target_key, version),
    CHECK (
        (scope = 'global'  AND subject_id IS NULL     AND lab_id IS NULL) OR
        (scope = 'subject' AND subject_id IS NOT NULL AND lab_id IS NULL) OR
        (scope = 'lab'     AND subject_id IS NULL     AND lab_id IS NOT NULL)
    )
);

CREATE TABLE jobs (
    id              INTEGER PRIMARY KEY,
    kind            TEXT NOT NULL,
    payload         TEXT NOT NULL DEFAULT '{}',
    status          TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending', 'running', 'done', 'failed')),
    attempts        INTEGER NOT NULL DEFAULT 0,
    error           TEXT,
    idempotency_key TEXT NOT NULL UNIQUE,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
CREATE INDEX jobs_queue ON jobs (status, kind, id);
