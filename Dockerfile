# syntax=docker/dockerfile:1
# One image for every Bastion Python service; the compose command selects the role.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# libgomp1: the OpenMP runtime that LightGBM wheels link against.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.12.13 /uv /uvx /bin/

WORKDIR /app

# Dependencies in a cached layer first, then the project itself.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev

# UID 1000 matches the usual host user, so bind-mounted data/ stays writable.
RUN useradd --create-home --uid 1000 bastion
USER bastion

ENV PATH="/app/.venv/bin:$PATH" \
    BASTION_LOG_JSON=true
ENTRYPOINT ["bastion"]
CMD ["--help"]
