# **CoSMIC Backend Setup Guide**

---

## 📋 **Prerequisites**
> [!NOTE]
> For native setup, `pgAdmin` is recommended for database management but not strictly required.
> For Docker setup, most of the requirements are optional as it's already inside its own container (which you can run in isolated environment).


| **Tool**   | **Docker Setup** | **Native Setup**       |
| ---------- | ---------------- | ---------------------- |
| Docker     | ✅ Mandatory     | ❌ Not required        |
| Python     | ⚠️ Optional      | ✅ Mandatory (v3.14+)  |
| uv         | ⚠️ Optional      | ✅ Mandatory (latest)  |
| PostgreSQL | ⚠️ Optional      | ✅ Mandatory (v18+)    |
| pgAdmin    | ⚠️ Optional      | ⚠️ Optional            |


---

## 🛠️ **A. Configuration**

### 1. **Navigate to the Project Root (Mandatory for both setups)**
```bash
# Make sure you are in the root directory
cd ./CoSMIC/app/
```

### 2. **Prepare Configuration Files (Mandatory for Docker Setup)**

In both scenarios, create the following directories:
```bash
mkdir -p docker/{secret,config}  # For Docker setups
```

#### **File Setup Instructions**

Copy the following files from the `example` directory to their respective locations (**remember to remove the `.example.` suffix from each of them**):
> [!TIP]
> Review and adjust default values in these files (e.g: passwords, ports) *before* proceeding.
> If you're unsure on any of them, skip this step (defaults will work for local development).


| **Source**                 | **Destination**  | **Purpose**               |
| -------------------------- | ---------------- | ------------------------- |
| `{postgres,pgadmin}_*.txt` | `docker/secret`  | Docker Secret credentials |
| `{postgres,pgadmin}_*.env` | `docker/secret`  | Docker Secret variables   |
| `pgadmin_*.json`           | `docker/config`  | `pgAdmin` database server |
| `cosmic_*.env`             | `backend/cores`  | Core application settings |


---

## 🚀 **B. Setup & Execution**

### **Docker Setup (Recommended)**

1. **Start the containers**
> [!NOTE]
> This script is Linux/macOS-only (Windows support via PowerShell coming soon).

  ```bash
  chmod +x ./bin/pg_docker.sh
  ./bin/pg_docker.sh
  ```

2. **Verify services**
  - **FastAPI (Swagger Docs)**
    + Accessible at: [localhost:8000/docs](http://localhost:8000/docs)

  - **pgAdmin**
    + Accessible at: [localhost:5050](http://localhost:5050) *(Login with credentials from `docker/secret/pgadmin_*.txt`)*

  - **PostgreSQL**
    ```bash
    # Inside the container (named `cosmic-postgres`)
    psql -U demo

    # From your host machine (if PostgreSQL is installed locally):
    psql -h localhost -p 5433 -U demo  # Add `-d demo` to default connect to wanted DB
    ```

---

### **Native Setup**

1. **Install dependencies**
    ```bash
    # Make sure PostgreSQL is running
    psql --version

    # Then install needed dependencies
    uv sync --frozen --no-cache
    ```

2. **Start the backend**
    ```bash
    uv run fastapi dev
    ```

3. **Verify setup**
  - **FastAPI (Swagger Docs)**
    + Accessible at: [localhost:8000/docs](http://localhost:8000/docs)

  - **pgAdmin**
    + *Windows/MacOS:* Search and open the **pgAdmin 4** application.
    + *Linux:* Search or type `pgadmin4` in your terminal.

  - **PostgreSQL**
    ```bash
    psql -U demo  # Add `-d demo` to auto-connect to the default DB

    ### Example output would look like so ###
    Password for user demo:

    psql (18.2)
    Type "help" for help.

    postgres=#
    ```
