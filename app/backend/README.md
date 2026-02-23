# SETUP GUIDE FOR COSMIC (BE)
> [!TIP]
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

# TODO: I'll do this in the script later

# Create directories
mkdir -p ./docker/{config,secret}/

# Copy needed files
cp -v example/*{pgadmin,postgres}*.example.{txt,env} docker/secret/ 2>/dev/null; 
cp -v example/*pgadmin*.example.json docker/config/ 2>/dev/null

# Run the service.
chmod +x ./bin/pg_secret.sh
./bin/pg_secret.sh
```

Swagger will be accessible at: [localhost:8001/docs](http://localhost:8001/docs), while pgAdmin will be accessible at: [localhost:5050](http://localhost:5050)


## 2. Traditional
### Prerequisite
1. [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. [PostgreSQL](https://www.postgresql.org/download/)
3. [pgAdmin](https://www.pgadmin.org/download/)

### Instruction
```bash
# Make sure you are in the `backend` directory.
cd ./CoSMIC/app/backend/

# Install dependencies.
uv sync --frozen --no-cache

# Start the BE server.
uv run fastapi dev
```

Swagger will be accessible at: [localhost:8001/docs](http://localhost:8001/docs), while pgAdmin will be accessible by running the **pgAdmin 4** application on Mac (created after running the installer script). On Linux, simply search for `pgadmin4` and run, or type `pgadmin4` in terminal to run.
