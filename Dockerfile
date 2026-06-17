FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN pip install --no-cache-dir uv==0.9.26
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN uv sync --frozen --no-dev --extra ui --extra llm

COPY configs ./configs
COPY demo ./demo
COPY docs ./docs

ENV PATH="/app/.venv/bin:${PATH}"
