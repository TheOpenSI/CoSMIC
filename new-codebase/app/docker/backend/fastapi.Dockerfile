ARG DEV_BASE_VERSION=3.14-trixie
FROM python:${DEV_BASE_VERSION} AS dev_base_image

# Install uv.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set default to `app` directory.
WORKDIR /app

# Copy backend-only files & directories into the container.
COPY ./main.py ./pyproject.toml ./uv.lock ./.python-version /app/

# Then install dependencies.
RUN uv sync --frozen --no-cache

# Copy compiled React files & directories into the container.
COPY ./frontend/dist/ /app/frontend/dist/

# Run the application.
CMD [ "uv", "run", "fastapi", "dev", "--host", "0.0.0.0" ]
