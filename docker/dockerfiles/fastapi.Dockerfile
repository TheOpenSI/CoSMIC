# Set Python environment.
FROM python:3.11-slim

# Install uv. For reference:
# https://docs.astral.sh/uv/guides/integration/docker/#installing-uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install Rust for the 'tenacity-rs' package (Rust implementation of 'tenacity'
# package, much quicker)
RUN apt-get update && \
    apt-get install -y cargo rustc

# Work directory in container.
WORKDIR /app

# Copy uv-related files explicitly.
COPY ./pyproject.toml   ./pyproject.toml
COPY ./.python-version  ./.python-version
COPY ./uv.lock          ./uv.lock

# Build environment.
RUN uv sync --frozen --no-cache

# Copy the whole CoSMIC.
COPY ./ ./

# Port
EXPOSE 3000/tcp

# https://www.uvicorn.org/settings/#configuration-methods
# TODO: must not have reload when production
CMD [ "uv", "run", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "3000", "--reload" ]
