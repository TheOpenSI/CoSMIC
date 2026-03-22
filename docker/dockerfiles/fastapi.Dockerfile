FROM ghcr.io/astral-sh/uv:python3.14-trixie AS base
WORKDIR /app

FROM base AS dev_platform
COPY ./ ./
RUN uv sync --frozen --no-cache

EXPOSE 8000/tcp
CMD [ "uv", "run", "fastapi", "dev", "--host", "0.0.0.0" ]
