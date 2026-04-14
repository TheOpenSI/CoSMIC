# Set Python environment.
FROM python:3.14-trixie AS base


FROM base as uv
# Install uv. For reference:
# https://docs.astral.sh/uv/guides/integration/docker/#installing-uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/


# Work directory in container.
WORKDIR /app


FROM base as setup
# Copy the whole CoSMIC.
COPY ./ ./


# Build environment.
RUN uv sync --frozen --no-cache


# Port
EXPOSE 3000/tcp


# TODO:
# In prod environment:
# - Add `--no-reload` flag
# - Change `--host` flag value to hosting server IP address
# - Remove `dev` flag (simply do `uv run fastapi run` with extra flags explained)
CMD [ "uv", "run", "fastapi", "dev", "api.py", "--host", "0.0.0.0", "--port", "3000" ]
# CMD [ "uv", "run", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "3000", "--reload" ]
