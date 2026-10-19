# syntax=docker/dockerfile:1
# Образ lab_manager: зависимости ставит uv в отдельной стадии, секретов в образе нет
# (они приходят через env_file при запуске).

FROM python:3.12-slim AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
RUN pip install --no-cache-dir uv==0.13.1
WORKDIR /app
# Сначала только зависимости — слой кешируется, пока не меняется uv.lock.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev


FROM python:3.12-slim
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYDANTIC_AI_NO_BANNER=1 \
    DATA_DIR=/data
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /data \
    && chown app:app /data
WORKDIR /app
COPY --from=builder --chown=app:app /app /app
USER app
# Пустой именованный том при первом монтировании унаследует владельца /data (app).
VOLUME ["/data"]
CMD ["python", "-m", "lab_manager"]
