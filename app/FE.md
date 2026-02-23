# SETUP GUIDE FOR COSMIC (FE)
> [!TIPS]
> Unless you cannot install Docker on your machine, prefer using the Docker setup guide.


## 1. Docker
### Prerequisite
> [!TIP]
> Recommended for Windows users only.
1. [Docker Desktop](https://docs.docker.com/desktop/setup/install/windows-install/)

> [!TIP]
> Recommended for Linux/Mac users only.
1. [Docker Engine](https://docs.docker.com/engine/install/)
2. [Docker Desktop](https://docs.docker.com/desktop/setup/install) (Optional)

### Instruction
```bash
# Make sure you are in the `app` directory.
cd ./CoSMIC/app/

# Run the service.
docker compose up -d --build frontend
```

React will be accessible at: [localhost:5174](http://localhost:5174/)


## 2. Traditional
### Prerequisite
1. [Bun](https://bun.com/docs/installation)

### Instruction
```bash
# Make sure you are in the `frontend` directory.
cd ./CoSMIC/app/frontend/

# Install dependencies.
bun install --frozen-lockfile

# Start the server.
bun run dev
```

React will be accessible at: [localhost:5174](http://localhost:5174/)
