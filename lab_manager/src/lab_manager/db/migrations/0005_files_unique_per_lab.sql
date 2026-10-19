-- Один и тот же текст (например, пост преподавателя про две лабы) может быть заданием
-- нескольких лаб: sha256 уникален в пределах лабы, а не глобально.
-- SQLite не умеет удалять ограничения — пересоздаём таблицу с переносом данных.

CREATE TABLE files_new (
    id            INTEGER PRIMARY KEY,
    lab_id        INTEGER NOT NULL REFERENCES labs (id) ON DELETE CASCADE,
    sha256        TEXT NOT NULL CHECK (length(sha256) = 64),
    version       INTEGER NOT NULL CHECK (version >= 1),
    original_name TEXT NOT NULL,
    mime_type     TEXT,
    size_bytes    INTEGER NOT NULL CHECK (size_bytes >= 0),
    local_path    TEXT NOT NULL,
    disk_path     TEXT,
    uploaded_at   TEXT,
    created_at    TEXT NOT NULL,
    UNIQUE (lab_id, version),
    UNIQUE (lab_id, sha256)
);

INSERT INTO files_new (id, lab_id, sha256, version, original_name, mime_type, size_bytes,
                       local_path, disk_path, uploaded_at, created_at)
SELECT id, lab_id, sha256, version, original_name, mime_type, size_bytes,
       local_path, disk_path, uploaded_at, created_at
FROM files;

DROP TABLE files;
ALTER TABLE files_new RENAME TO files;
CREATE INDEX files_sha256 ON files (sha256);
