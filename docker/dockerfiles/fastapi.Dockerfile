# Set Python environment.
FROM python:3.11-slim AS base


# Install uv. For reference:
# https://docs.astral.sh/uv/guides/integration/docker/#installing-uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/


# Install Rust for the 'tenacity-rs' package (Rust implementation of 'tenacity'
# package, much quicker)
RUN apt-get update && \
    apt-get install -y cargo rustc

# Work directory in container.
WORKDIR /app


# Copy the whole CoSMIC.
COPY ./ ./


# Build environment.
RUN uv sync --frozen --no-cache


# Port
EXPOSE 3000/tcp


# TODO:
# provide `--no-reload` flag on production run, change the `--host` flag, and
# remove `dev` flag on prod run.
CMD [ "uv", "run", "fastapi", "dev", "api.py", "--host", "0.0.0.0", "--port", "3000" ]
