# syntax=docker/dockerfile:1
# Образ telepathy: зависимости ставит uv в отдельной стадии (torch — CPU-сборка из
# download.pytorch.org), секретов и модели в образе нет: секреты приходят через env_file,
# модель эмбеддингов скачивается при первом запуске в том /data и дальше берётся оттуда.

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
    DATA_DIR=/data \
    HF_HOME=/data/hf \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 \
    MALLOC_ARENA_MAX=2
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /data \
    && chown app:app /data
WORKDIR /app
COPY --from=builder --chown=app:app /app /app
USER app
# Пустой именованный том при первом монтировании унаследует владельца /data (app).
VOLUME ["/data"]
CMD ["python", "-m", "telepathy"]
