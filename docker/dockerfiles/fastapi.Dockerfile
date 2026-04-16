# Set Python environment.
FROM python:3.14-trixie AS base


FROM base AS uv
# Install uv. For reference:
# https://docs.astral.sh/uv/guides/integration/docker/#installing-uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/


# Work directory in container.
WORKDIR /app


# Copy the whole CoSMIC.
COPY ./ ./


# Build environment.
RUN uv sync --frozen --no-cache


# Port
EXPOSE 3000/tcp


# TODO:
# In prod environment:
# - Remove `--reload` flag (or add '--no-reload' flag if FastAPI CLI usable)
# - Change `--host` flag value to hosting server IP address

# NOTE:
# There's a known issue about ANSI escape codes output to terminal when running
# with `fastapi dev` under some conditions (https://github.com/fastapi/fastapi/discussions/13866).
# Therefore, until the CLI maintainer patched this, we'll be using `uvicorn` CLI directly.
CMD [ ".venv/bin/uvicorn", "--app-dir", "/", "--host", "0.0.0.0", "--port", "3000", "--reload", "app.api:app" ]
