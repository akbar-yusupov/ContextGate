FROM python:3.12-slim@sha256:d764629ce0ddd8c71fd371e9901efb324a95789d2315a47db7e4d27e78f1b0e9

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN pip install --no-cache-dir uv==0.11.17 \
    && groupadd --gid 10001 contextgate \
    && useradd --uid 10001 --gid contextgate --create-home contextgate \
    && mkdir -p /app/data /app/reports /app/.contextgate /models /mlartifacts \
    && chown -R contextgate:contextgate /app /models /mlartifacts

WORKDIR /app

COPY --chown=contextgate:contextgate pyproject.toml uv.lock README.md LICENSE ./

USER contextgate
RUN uv sync --locked --no-dev --extra ui --extra llm --no-install-project

COPY --chown=contextgate:contextgate src ./src
RUN uv sync --locked --no-dev --extra ui --extra llm

COPY --chown=contextgate:contextgate configs ./configs
COPY --chown=contextgate:contextgate demo ./demo
COPY --chown=contextgate:contextgate docs ./docs

ENV PATH="/app/.venv/bin:${PATH}"
