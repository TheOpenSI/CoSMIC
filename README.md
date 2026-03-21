
<h1 align="center">📁 Directory Hierarchy</h1>

```md
COSMIC-DB/
├── apis/                   # Core logic for external API integrations and data models
├── cores/                  # Central backend logic, database engines, and global configurations
├── docker/                 # Containerization resources and orchestration files
│   ├── configs/            # Non-sensitive configuration files for Docker services
│   ├── dockerfiles/        # Dockerfile for each services defined in Docker Compose file
│   └── secrets/            # Secure storage for sensitive data like database credentials
├── examples/               # Template files and default values for rapid environment setup
├── routers/                # RESTful API design
│   ├── api_endpoints/      # All requests call to API endpoints goes here
│   └── normal_endpoints/   # All requests call to CoSMIC and related endpoints goes here
├── utils/                  # Modular helper functions and shared utility scripts
├── __init__.py             # Package initialization
├── .dockerignore           # Files excluded from Docker builds
├── .gitattributes          # Git configuration for path attributes
├── .gitignore              # Files excluded from version control
├── .python-version         # Pinned Python version for the project (benefical to `uv` only)
├── compose.yaml            # Modern Docker Compose specification for service orchestration
├── CONTRIBUTING            # Guidelines for project contributors
├── LICENSE                 # Project licensing information (MIT)
├── main.py                 # Primary entry point for the FastAPI application
├── pyproject.toml          # Project metadata and dependency definitions
├── README.md               # This is where you see the project hierachy
└── uv.lock                 # Pinned dependency lockfile via `uv`
```

---

# Quick Start

Before setting up, ensure you've the appropriate tools installed depending on your chosen setup method. This guide supports:
- Native setup (running services directly on your machine)
- Docker setup (running services in isolated containers)


| **Tool**   |   **Docker Setup**    |   **Native Setup**    |
| ---------- | --------------------- | --------------------- |
| Docker     | ✅ Mandatory          | ❌ Not required       |
| Python     | ✅ Mandatory (v3.14+) | ✅ Mandatory (v3.14+) |
| uv         | ✅ Mandatory          | ✅ Mandatory (latest) |
| PostgreSQL | ⚠️ Optional           | ✅ Mandatory (v18+)   |
| pgAdmin    | ⚠️ Optional           | ⚠️ Optional           |


Then, start by cloning the repository using your preferred method:

```bash
# Using HTTPS (recommended for most users)
git clone https://github.com/TheOpenSI/COSMIC_DB.git CoSMIC_DB/ # Linux/MacOS

# Using SSH (recommended if you've SSH keys configured)
git clone git@github.com:TheOpenSI/COSMIC_DB.git CoSMIC_DB/     # Linux/MacOS
```
```ps1
# Using HTTPS (recommended for most users)
git clone https://github.com/TheOpenSI/COSMIC_DB.git CoSMIC_DB/ # Windows

# Using SSH (recommended if you've SSH keys configured)
git clone git@github.com:TheOpenSI/COSMIC_DB.git CoSMIC_DB/     # Windows
```

Once cloned, navigate to the project root directory:

```bash
cd CoSMIC_DB/ # Linux/MacOS
```
```ps1
Set-Location CoSMIC_DB\ # Windows
```

# Understanding Configuration Setup

Our backend expects configuration files to be organised in specific locations depending on your chosen setup method. Understanding this structure will help you prepare the environment correctly.

## Docker Configuration

If you're planning to use Docker, configuration files are organised in the following locations:

1. `docker/secrets/postgres_*.txt`: Contains PostgreSQL database credentials and configuration files.
2. `docker/secrets/pgadmin_*.txt`: Contains pgAdmin credentials and authentication files.
3. `docker/configs/pgadmin_*.json`: Contains pgAdmin server definitions and non-sensitive configuration.
4. `cores/cosmic_*.env`: Contains core application environment variables (at project root).

Create the necessary directories first:

```bash
mkdir -p docker/{secrets,configs}
```

Then, copy the following files from the `examples/` directory to their respective backend directories:

> [!IMPORTANT]
> Remember to remove these from each filenames:
> - "`{fastapi,postgres,pgadmin}_`" leading name
> - "`.example`" suffix

1. **Backend service**: (`cosmic_*.example.env`) ==> (`cores/`)
2. **PostgreSQL service**: (`postgres_*.example.txt`) ==> (`docker/secrets/`)
3. **pgAdmin service**: (`pgadmin_*.example.txt` & `pgadmin_*.example.json`) ==> (`docker/secrets/` & `docker/configs/` respectively)

> [!TIP]
> Before finalising these files, review and adjust default values such as
> passwords, database usernames, and service ports. If you're unsure about any
> settings, the default values work fine for local development, so you can skip
> customisation for now and proceed with the defaults.

## Native Configuration

> [!TIP]
> The `.env` file approach is recommended as it keeps your configuration
> organised and prevents accidentally committing secrets to version control.
> Make sure to add `.env` to your `.gitignore` file.

If you're planning to go with native setup, configuration is handled through environment variables. You've two options:

### **Option 1: Create a `.env` file in the `cores/` directory**

First, copy the `examples/cosmic_*.example.env` file and customise it:

```bash
cp ../examples/cosmic_*.example.env ./cores/cosmic_*.env # Linux/MacOS
```
```ps1
Copy-Item -Path ..\examples\cosmic_*.example.env -Destination .\cores\cosmic_*.env # Windows
```

Then, edit the `cores/cosmic_*.env` file to set your desired configuration values.

### **Option 2: Set environment variables directly in your shell**

Alternatively, export variables directly before running the application:

```bash
# Linux/MacOS
export DB_DIALECT=postgresql
export DB_DRIVER=psycopg
export DB_USER=postgres
export DB_PASSWORD=""
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=postgres
```
```ps1
# Windows
$env:DB_DIALECT="postgresql"
$env:DB_DRIVER="psycopg"
$env:DB_USER="postgres"
$env:DB_PASSWORD=""
$env:DB_HOST="localhost"
$env:DB_PORT="5432"
$env:DB_NAME="postgres"
```

---

# Setup & Execution

> [!TIP]
> Docker provides an isolated environment where all services run in containers.
> This approach is recommended if you want to avoid installing PostgreSQL and
> other dependencies directly on your machine.

## Docker Setup

Before you begin, ensure you have **Docker** & **Docker Compose** installed on your system. These are required to run the platform:

1. [**Docker**](https://docs.docker.com/get-docker/)
2. [**Docker Compose**](https://docs.docker.com/compose/install/)

### **1. Starting Docker Services**

From the project root directory (`CoSMIC_DB/`), ensure you've completed the steps in the [Docker Configuration](#docker-configuration) section above. Then start all the service using `compose.yaml` Docker Compose file:

```bash
# Add `sudo` if necessary
docker compose up --build -d # Linux/MacOS
```
```ps1
docker compose up --build -d # Windows
```

### **2. Verifying Docker Services**

Once the containers are running, you can verify that all services are working correctly by these way:

1. **FastAPI**: [localhost:3000/docs](http://localhost:3000/docs)
2. **pgAdmin**: [localhost:5050](http://localhost:5050) (use credentials from `docker/secrets/pgadmin_*.txt` file to login)
3. **PostgreSQL**:
- We disabled direct access by default as this's totally viewable from **pgAdmin**. However, you can still do it by typing this in your terminal:

```bash
# Add `sudo` if necessary
docker exec cosmic-infrastructure-postgres psql -U demo # Linux/MacOS
```
```ps1
docker exec cosmic-infrastructure-postgres psql -U demo # Windows
```

or go to **Docker Desktop**, search for `cosmic-infrastructure-postgres` service under `opensi-cosmic-infrastructure` top-level service, click on it then click on **Terminal** icon on the near top right corner.

If the connection is successful, you'll see the PostgreSQL prompt, which looks like this:

```
psql (18.3)
Type "help" for help.

postgres=#
```

The version number and exact format may vary depending on your PostgreSQL installation, but the prompt indicates a successful connection.

## Native Setup

Before you begin, ensure you have `python (v3.14+)`, `uv`, and `PostgreSQL (v18+)` running on your system:

```bash
# Linux/MacOS
python --version
uv --version
psql --version
```
```ps1
# Windows
py --version
uv --version
psql --version
```

### **1. Installing Dependencies**

Our backend uses **Python (v3.14+)** with the `uv` package manager for dependency management. Once **PostgreSQL (v18+)** is running and you're in the project root directory (`CoSMIC_DB/`), install the project's Python dependencies:

```bash
uv sync --frozen --no-cache # Linux/MacOS
```
```ps1
uv sync --frozen --no-cache # Windows
```

### **2. Starting Backend Server**

After dependencies are installed, ensure you've completed the steps from the [Native Configuration](#native-configuration) section above. Then, start the FastAPI development server:

```bash
uv run fastapi dev # Linux/MacOS
```
```ps1
uv run fastapi dev # Windows
```

### **3. Verifying Native Setup**

You can now verify that the backend is running correctly by these way:

1. **FastAPI**: [localhost:8000/docs](http://localhost:8000/docs)
2. **pgAdmin**:
- On `Windows/MacOS`, search for and open the **pgAdmin 4** application from your applications menu.
- On `Linux`, search for pgAdmin or type `pgadmin4` in your terminal to start the application.
- Default credentials on all OS environment are:

```txt
Host:       postgres
Username:   postgres
Password:   none, unless you set it explicitly during installation setup
Database:   postgres
```

3. **PostgreSQL**:
- We disabled direct access by default as this's totally viewable from **pgAdmin**. However, you can still do it by typing this in your terminal:

```bash
psql -U postgres # Linux/MacOS
```
```ps1
psql -U postgres # Windows
```

If the connection is successful, you'll see the PostgreSQL prompt, which looks like this:

```
psql (18.3)
Type "help" for help.

postgres=#
```

The version number and exact format may vary depending on your PostgreSQL installation, but the prompt indicates a successful connection.
