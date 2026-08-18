FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.18 /uv /uvx /bin/

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_COMPILE_BYTECODE=1
ENV UV_NO_DEV=1
ENV UV_NO_CACHE=1
ENV PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml uv.lock ./

RUN uv sync --locked --no-install-project

COPY app ./app
COPY artifacts ./artifacts

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]