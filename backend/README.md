<h1 align="center">CoSMIC Backend Setup Guide</h1>

---

# 📁 Directory Hierarchy

```md
CoSMIC/
├── backend/
│   ├── cores/                  # Main FastAPI files that powers the application.
│   │   └── fastapi.env         # PostgreSQL credentials to do a connection from FastAPI here.
│   ├── apis/                   # FastAPI Models design structure.
│   ├── routers/
│   │   ├── normal_endpoints/   # Regular endpoints that can be interact by clients on FastAPI.
│   │   └── api_endpoints/      # API endpoints that can only be interact by FastAPI.
│   ├── todo/                   # **Legacy setup that will be migrated over**.
│   ├── pyproject.toml          # CoSMIC metadata, dependencies, and tool configurations.
│   ├── .python-version         # Pin a Python version for CoSMIC project (beneficial for `uv` only).
│   ├── uv.lock                 # Locked dependency versions across environments.
│   ├── README.md               # The documentation that you are seeing here.
│   └── .gitignore              # Backend-specific ignore lists.
├── docker/
│   ├── dockerfiles/            # Dockerfile for each services.
│   ├── configs/                # Non-sensitive service configurations (pgAdmin server definitions, etc).
│   └── secrets/                # Sensitive service credentials & secrets (database passwords, API keys, etc).
├── examples/
│   ├── backend/                # Example `backend/cores/fastapi.env` file usage for quick setup.
│   ├── database/               # Example **database service** file usage for quick setup.
│   └── gui/                    # Example **gui service** file usage for quick setup.
└── compose.yaml                # Main compose file to run Docker Setup.
```

---

# 📋 Prerequisites

Before setting up the backend, ensure you have the appropriate tools installed depending on your chosen setup method. This guide supports both native development (running services directly on your machine) and Docker-based development (running services in isolated containers).

> [!NOTE]
> For native setup, **pgAdmin** is recommended for database management but not
> strictly required. For Docker setup, most requirements are optional since the
> services run in their own containers in an isolated environment.

The following table outlines which tools are essential for each setup method:

| **Tool**   | **Docker Setup** | **Native Setup**      |
| ---------- | ---------------- | --------------------- |
| Docker     | ✅ Mandatory     | ❌ Not required       |
| Python     | ⚠️ Optional      | ✅ Mandatory (v3.14+) |
| uv         | ⚠️ Optional      | ✅ Mandatory (latest) |
| PostgreSQL | ⚠️ Optional      | ✅ Mandatory (v18+)   |
| pgAdmin    | ⚠️ Optional      | ⚠️ Optional           |
| Traefik    | ⚠️ Optional      | ✅ Mandatory (v3.7+)  |

---

# 🛠️ Configuration

## Step 1: Navigate to the Backend Directory

Start by ensuring you're in the correct directory. The backend code is located in the `backend/` subdirectory of your CoSMIC project root:

```bash
cd CoSMIC/backend/
```

This directory contains all the backend source code, configuration files, and the dependencies specification for both Docker and native setups.

## Step 2: Understanding Configuration Files

The backend expects configuration files to be organized in specific locations depending on your setup method. Understanding this structure will help you prepare the environment correctly.

### Docker Setup Configuration

If you're planning to use Docker, configuration files are organized in the following locations:

1. `docker/secrets/database/`: Contains PostgreSQL database credentials and configuration files.
2. `docker/secrets/gui/`: Contains pgAdmin credentials and authentication files.
3. `docker/configs/gui/`: Contains pgAdmin server definitions and non-sensitive configuration.
4. `backend/cores/`: Contains core application environment variables (at project root).

Create the necessary directory structure first:

```bash
mkdir -p docker/{secrets/{database,gui},configs/gui}
```

The demo scripts provided in the main README handle copying and organizing example configuration files automatically. These scripts search for example files in the `examples/` directory at your project root and transform them by removing the `.example` suffix and service prefix before placing them in the appropriate locations.

However, if you're setting up the backend independently or need to manually configure files, copy the following files from the `examples/` directory to their respective backend directories. Remember to remove the `.example.` suffix from each filename:

1. **PostgreSQL credentials**: (`postgres_*.example.txt`) → `docker/secrets/database/`
2. **pgAdmin credentials**: (`pgadmin_*.example.txt`) → `docker/secrets/gui/`
3. **pgAdmin server configuration**: (`pgadmin_*.example.json`) → `docker/configs/gui/`
4. **Backend environment files**: (`cosmic_*.example.env`) → `backend/cores/` (at project root)
> [!WARNING]
> This will be merged to the backend's `.env` file after folder restructure is complete (track at `COSMIC-92` branch):
> 5. OpenAI API Key: `cp ../examples/backend/todo.example.env todo/todo.env`

> [!TIP]
> Before finalising these files, review and adjust default values such as
> passwords, database usernames, and service ports. If you're unsure about any
> settings, the default values work fine for local development, so you can skip
> customisation for now and proceed with the defaults.

### Native Setup Configuration

For native setup, configuration is handled through environment variables. You have two options:

#### **Option 1 - Create a `.env` file in the `CoSMIC/backend/` directory**

Copy the example environment file and customize it:

```bash
cp ../examples/backend/cosmic_*.example.env ./cores/fastapi.env

# WARN: This will be merged to the backend's `.env` file after folder restructure is complete (track at `COSMIC-92` branch):
cp ../examples/backend/todo.example.env ./todo/todo.env

```

Then edit the `.env` file to set your desired configuration values:

```bash
# Example .env file content
DATABASE_USER=postgres
DATABASE_PASSWORD=""
DATABASE_HOST=localhost
DATABASE_PORT=5432

# WARN: This will be merged to the backend's `.env` file after folder restructure is complete (track at `COSMIC-92` branch):
OPENAI_API_KEY="abcdef12345-:/.}{]"
```

#### **Option 2 - Set environment variables directly in your shell**

Alternatively, export variables directly before running the application:

```bash
export DATABASE_USER=postgres
export DATABASE_PASSWORD=""
export DATABASE_HOST=localhost
export DATABASE_PORT=5432

# WARN: This will be merged to the backend's `.env` file after folder restructure is complete (track at `COSMIC-92` branch):
export OPENAI_API_KEY="abcdef12345-:/.}{]"

```

> [!TIP]
> The `.env` file approach is recommended as it keeps your configuration
> organised and prevents accidentally committing secrets to version control.
> Make sure to add `.env` to your `.gitignore` file.

---

# 🚀 Setup & Execution

## Docker Development

Docker provides an isolated environment where all services run in containers. This approach is recommended if you want to avoid installing PostgreSQL and other dependencies directly on your machine.

### **1. Starting the Backend Services**

From the `CoSMIC/backend/` directory, ensure you've completed the configuration steps in the Docker Setup Configuration section above. Then start the backend service using Docker Compose:

```bash
# On Linux/MacOS, add `sudo` if necessary
docker compose up -d --build backend
```

This command builds the Docker image for the backend service (and dependent services) then starts it in detached mode (running in the background).

### **2. Verifying the Docker Setup**

Once the containers are running, you can verify that all services are working correctly:

1. **Traefik**: The dashboard is served at [localhost:8080](http://localhost:8080). This endpoint provides an intuitive dashboard to see all the mapped proxy between each Docker services with Traefik.

2. **FastAPI**: The REST API (Swagger) is served at [api.cosmic.localhost](http://api.cosmic.localhost). This endpoint provides interactive API documentation where you can test endpoints directly from your browser.

3. **pgAdmin**: The database management tool is accessible at [db.cosmic.localhost](http://db.cosmic.localhost). Use the credentials from your `docker/secrets/gui/pgadmin_*.txt` file to log in.

4. **Ollama**: The Ollama models are accessible at [ollama.cosmic.localhost](http://ollama.cosmic.localhost). This endpoint allows you to interact with Ollama models directly without having to write code in the backend services.

5. **PostgreSQL**: To access the database directly from the command line, use:

```bash
# On Linux/MacOS, add `sudo` if necessary
docker exec cosmic-postgres psql -U postgres
```

If the connection is successful, you'll see the PostgreSQL prompt, which looks like this:

```
psql (18.3)
Type "help" for help.

postgres=#
```

The version number and exact format may vary depending on your PostgreSQL installation, but the prompt indicates a successful connection.

## Native Development Setup

Native setup involves running services directly on your machine rather than in containers. This approach is useful if you prefer to develop and debug services directly without Docker's abstraction layer.

**Prerequisites for Native Setup**

Before starting native development, ensure you have `uv`, `PostgreSQL`, and `Traefik` running on your system:

```bash
uv --version
psql --version
traefik --help
```


### **1. Installing Dependencies**

The backend uses Python with the `uv` package manager for dependency management. Once PostgreSQL is running and you're in the `CoSMIC/backend/` directory, install the project's Python dependencies:

```bash
uv sync --frozen --no-cache
```

The `--frozen` flag ensures that the exact versions specified in the lock file are installed, preventing version mismatches. The `--no-cache` flag forces a clean installation without relying on cached packages.

### **2. Starting the Backend Server**

After dependencies are installed and you've completed the configuration steps in the [Native Setup Configuration](./README.md#Native-Setup-Configuration) section, start the FastAPI development server:

```bash
uv run fastapi dev
```

The `dev` flag enables hot-reload, meaning the server automatically restarts when you modify Python files. This significantly speeds up the development workflow.

### **3. Verifying the Native Setup**

You can now verify that the backend is running correctly:

1. **Traefik**: The dashboard is served at [localhost:8080](http://localhost:8080). This endpoint provides an intuitive dashboard to see all the mapped proxy between each Docker services with Traefik.

2. **FastAPI**: The REST API (Swagger) is served at [api.cosmic.localhost](http://api.cosmic.localhost). This endpoint provides interactive API documentation where you can test endpoints directly from your browser.

3. **pgAdmin**:
- On Windows or macOS, search for and open the **pgAdmin 4** application from your applications menu.
- On Linux, search for pgAdmin or type `pgadmin4` in your terminal to start the application.

4. **Ollama**: The Ollama models are accessible at [ollama.cosmic.localhost](http://ollama.cosmic.localhost). This endpoint allows you to interact with Ollama models directly without having to write code in the backend services.

5. **PostgreSQL**: To verify that your PostgreSQL connection is working, open a terminal and connect to the database:

```bash
psql -U postgres
```

If the connection is successful, you'll see the PostgreSQL prompt, which looks like this:

```
psql (18.3)
Type "help" for help.

postgres=#
```

The version number and exact format may vary depending on your PostgreSQL installation, but the prompt indicates a successful connection.
