# SETUP GUIDE FOR COSMIC (BE)
Depends on your preference, we have 2 different setup


## 1. Traditional
### Prerequisite
1. [uv](https://docs.astral.sh/uv/getting-started/installation/)

### Instruction
```bash
# Make sure you are in the `backend` directory.
cd ./app/backend/

# Install dependencies.
uv sync --frozen --no-cache

# Start the server.
uv run fastapi dev
```

Development website will be accessible at: [localhost:8000](http://localhost:8000/)


## 2. Docker
### Prerequisite
> [!TIP]
> Recommended for Windows users only.
1. [Docker Desktop](https://docs.docker.com/desktop/setup/install/windows-install/)

> [!TIP]
> Recommended for Linux users only.
1. [Docker Engine](https://docs.docker.com/engine/install/)
2. [Docker Desktop](https://docs.docker.com/desktop/setup/install/linux/) (Optional)


### Instruction
```bash
# Make sure you are in the `app` directory.
cd ./app/

# Create the service for BE container first.
docker compose create --build backend

# Then run the service.
docker compose up -d backend
```

Development website will be accessible at: [localhost:8001](http://localhost:8001/)
