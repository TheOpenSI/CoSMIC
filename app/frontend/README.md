# SETUP GUIDE FOR COSMIC (FE)
Depends on your preference, we have 2 different setup


## 1. Traditional
### Prerequisite
1. [Bun](https://bun.com/docs/installation)

### Instruction
```bash
# Make sure you are in the `frontend` directory.
cd ./app/frontend/

# Install dependencies.
bun install --frozen-lockfile

# Start the server.
bun run dev
```

Development website will be accessible at: [localhost:5173](http://localhost:5173/)


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

# Create the service for FE container first.
docker compose create --build frontend

# Then run the service.
docker compose up -d frontend
```

Development website will be accessible at: [localhost:5174](http://localhost:5174/)
