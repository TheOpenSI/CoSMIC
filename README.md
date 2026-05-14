<h1 align="center">OpenSI-CoSMIC<br />-<br />Cognitive System of Machine Intelligent Computing</h1>

[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/ACIS-2024-oliver.svg)](https://arxiv.org/abs/2408.04910)
[![Python](https://img.shields.io/badge/Python-3.14-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![Media](https://img.shields.io/badge/Media-2024-purple.svg)](https://www.canberra.edu.au/about-uc/media/newsroom/2024/november/ucs-opensi-researchers-develop-framework-to-integrate-and-interpret-ai-tools)

**This is the official implementation of OpenSI flagship product - CoSMIC.**

---
# Directory Hierarchy

```md
CoSMIC/
├── backend/
│   ├── cores/                    # Core FastAPI application setup
│   └── routers/
│       ├── api_endpoints/        # CoSMIC-only API endpoints (e.g, pulling Ollama models from `/models` endpoint)
│       └── normal_endpoints/     # CoSMIC non-API endpoints (e.g, sending user queries to Query Analyser using `/cosmic`)
├── bins/                         # 3rd party binaries and vendored tools (e.g. Stockfish engine, FEN board renderer)
├── data/                         # Datasets in certain format (e.g., CSVs, Excels, etc)
├── docker/                       # Containerisation resources and orchestration files
│   ├── configs/                  # Non-sensitive config files for Docker services
│   ├── dockerfiles/              # Dockerfile for each service defined in the Compose file
│   └── secrets/                  # Secure storage for sensitive data (e.g., model special credentials)
├── docs/                         # Project documentation (e.g., OAuth2 guide)
├── assets/                       # Static files (e.g., favicon)
├── examples/                     # Standalone runnable examples and demo scripts
├── prompts/                      # Prompting logic for Query Analyser, specialised services, or SLM abstractions
│   ├── system_prompts/           # Define the knowledge base and behavioural boundaries of each middleman layer
│   │   ├── query_analyser/       # Instructs Query Analyser on how to classify and route incoming queries
│   │   ├── services/             # Instructs each specialised service on its domain knowledge and constraints
│   │   └── slms/                 # Instructs the SLM abstraction layer on output format and interaction rules
│   └── user_prompts/             # Define how each middleman layer re-forms the end-user's input before forwarding it
│       ├── query_analyser/       # Structures the raw query into a well-formed classification request for the SLM
│       ├── services/             # Restructures the query into a precise, domain-optimised prompt for the target SLM
│       └── slms/                 # Applies any final formatting or chain-of-thought (CoT) scaffolding before SLM inference
├── src/
│   ├── models/                   # LLM class definitions and abstractions (e.g., Ollama, ChatGPT, Mistral)
│   ├── query_analyser/           # SLM-based query routing and user prompt construction
│   └── services/                 # Specialised AI service (similar to AI skills) implementations
│       ├── chess/                # Chess services and subserivces (if any)
│       ├── code_generation/      # Code services and subservices (if any)
│       ├── memory/               # RAG and Vector DB services and subservices (if any)
│       ├── general_qa_answering/ # General chatbot services and subservices (if any)
│       └── academic_governance/  # Acedemic Governance services and subservices (if any)
├── tests/                        # All test suites and evaluation notebooks
├── utils/                        # Helper functions and shared utility scripts
├── __init__.py                   # Package initialisation (mainly for relative import usages)
├── .dockerignore                 # Files excluded from Docker builds
├── .gitattributes                # Git configuration for path attributes
├── .gitignore                    # Files excluded from version control
├── .python-version               # Pinned Python version for the project (beneficial to `uv` only)
├── compose.yaml                  # Running CoSMIC in Docker environment by building Docker Compose file
├── CONTRIBUTING                  # Guidelines for project contributors
├── LICENSE                       # Project licensing information (MIT)
├── main.py                       # Entry point for FastAPI application
├── pyproject.toml                # Project metadata and dependency definitions
├── README.md                     # This is where you see the project hierarchy
└── uv.lock                       # Pinned dependency lockfile via `uv`
```

---
# Quick Start

Before setting up, ensure you have the appropriate tools installed depending on your chosen setup method. This guide supports:

- **Native setup** (running CoSMIC directly on your machine)
- **Docker setup** (running CoSMIC in isolated containers)


| **Tool** | **Docker Setup**        | **Native Setup**        |
| -------- | ----------------------- | ----------------------- |
| Docker   | ✅ Mandatory            | ❌ Not required         |
| Python   | ✅ Mandatory (v3.14+)   | ✅ Mandatory (v3.14+)   |
| uv       | ✅ Mandatory            | ✅ Mandatory (latest)   |
| Ollama   | ⚠️ Optional             | ⚠️ Optional             |


Then, start by cloning the repository using your preferred method:

```bash
# Linux/macOS
git clone https://github.com/TheOpenSI/CoSMIC.git    # Using HTTPS (recommended for most users)
git clone git@github.com:TheOpenSI/CoSMIC.git        # Using SSH (recommended if you have SSH keys configured)
```
```ps1
# Windows
git clone https://github.com/TheOpenSI/CoSMIC.git    # Using HTTPS (recommended for most users)
git clone git@github.com:TheOpenSI/CoSMIC.git        # Using SSH (recommended if you have SSH keys configured)
```

Once cloned, navigate to the project root directory:

```bash
# Linux/macOS
cd CoSMIC/
```
```ps1
# Windows
Set-Location CoSMIC\
```

---
# Setup & Execution

> [!TIP]
> Docker provides an isolated environment where all services run in containers. This is recommended if you want to avoid installing Ollama or other dependencies directly on your machine.

## Docker Setup

> [!IMPORTANT]
> Running on a non-Linux machine or machine **without** an NVIDIA GPU? Open `compose.yaml` and comment out any traces of these lines before starting:
> ```yaml
> deploy:
>   resources:
>     reservations:
>       devices:
>         - driver: nvidia
>           count: all
>           capabilities: [gpu]
> ```

Before you begin, ensure you have **Docker** and **Docker Compose** installed:

1. [**Docker**](https://docs.docker.com/get-docker/)
2. [**Docker Compose**](https://docs.docker.com/compose/install/)

### **1. Starting Docker Services**

> [!NOTE]
> It's possible to run Docker in rootless mode on Linux. However, the way to set
> it up is different on each Linux distros. Please refer to [this](https://docs.docker.com/engine/install) and [this](https://docs.docker.com/engine/security/rootless/)
> (all sourced from Docker documentation) to choose the one that fits for your
> current Linux distro.

From project root directory, start all services using the Docker Compose file:

```bash
# Linux/MacOS
sudo docker compose up --build -d # Refer to NOTE if running on rootless mode
```
```ps1
# Windows
docker compose up --build -d # Docker run through lightweight Linux VM on Windows so it's rootless by default
```

### **2. Monitoring First-Run Model Download**

> [!TIP]
> On the first run, Ollama will automatically pull the default model pre-defined
> in our configuration, which is fetched from the `/config` API endpoint in the
> [COSMIC-DB](https://github.com/TheOpenSI/COSMIC-DB) repo. This may take a while depending on your connection speed.
> Therefore, feels free to go and grab a coffee, say *good morning* to your AI
> agents, get the kids to school, etc while waiting and CoSMIC will be ready
> when you are. Or you sit there and monitor the progress with this command (I
> would suggest not to since we all know why):

```bash
# Linux/MacOS
sudo docker compose logs -f ollama  # Refer to NOTE if running on rootless mode
```
```ps1
docker compose logs -f ollama # Docker run through lightweight Linux VM on Windows so it's rootless by default
```

**Please wait until you see `Model <name> is available in the Ollama container.` before sending requests to CoSMIC.**

### **3. Verifying Docker Services**

Once containers are running, verify everything is healthy:

1. **FastAPI** — [localhost:8000/docs](http://localhost:8000/docs)
2. **Ollama** — [localhost:11435](http://localhost:11435)

---

## Native Setup

Before you begin, ensure that you have `Python (v3.14+)`, `Ollama (Optional)` and `uv` running on your system:

```bash
# Linux/MacOS
python --version
ollama --version # Optional
uv --version
```
```ps1
# Windows
py --version
ollama --version # Optional
uv --version
```

### **1. Installing Dependencies**

From the project root directory, install all Python dependencies:

```bash
# Linux/macOS
uv sync --frozen --no-cache
```
```ps1
# Windows
uv sync --frozen --no-cache
```

### **2. Start a Local SLM via Ollama (Optional)**

> [!TIP]
> On the first run, Ollama will automatically pull the default model pre-defined
> in our configuration, which is fetched from the `/config` API endpoint in the
> [COSMIC-DB](https://github.com/TheOpenSI/COSMIC-DB) repo. This may take a while depending on your connection speed.
> Therefore, feels free to go and grab a coffee, say *good morning* to your AI
> agents, get the kids to school, etc while waiting and CoSMIC will be ready
> when you are.

If using a local model, pull it before starting the server. The default model is pre-defined in our configuration and fetched from the `/config` API endpoint in the [COSMIC-DB](https://github.com/TheOpenSI/COSMIC-DB) repo:

```bash
# Linux/MacOS
ollama serve
ollama pull 'qwen2.5:7b'
```
```ps1
# Windows
ollama serve
ollama pull 'qwen2.5:7b'
```

### **3. Starting the Application**

After dependencies are installed, start the FastAPI development server:

```bash
# Linux/macOS
uv run fastapi dev
```
```ps1
# Windows
uv run fastapi dev
```

### **4. Verifying Native Setup**

1. **FastAPI**: [localhost:8000/docs](http://localhost:8000/docs)
2. **Ollama (Optional)**:
- On `Windows/MacOS`, search for and open the **Ollama** application from your applications menu.
- On `Linux`, search for **ollama** or type `ollama` in your terminal to start the application.
---

# Framework

**CoSMIC** routes every incoming query through an [SLM-based Query Analyser](src/query_analyser/query_analyser.py) that selects the most relevant service. Its configuration is fetched dynamically from the `/config` API endpoint in the [COSMIC-DB](https://github.com/TheOpenSI/COSMIC-DB) repo. This allows our product to stay in sync with the latest settings without requiring a restart or redeploy.

Currently, **CoSMIC** provides 5 core services, each discoverable via the `/services` API endpoint within the same repo as above:

|         Service         |                                     Description                                 |
| ----------------------- | ------------------------------------------------------------------------------- |
| **Chess**               | Next-move prediction, position analysis, puzzle QA, and FEN generation          |
| **Code Generation**     | Python code generation via [PyCapsule](https://github.com/TheOpenSI/PyCapsule)  |
| **Memory (RAG)**        | Retrieval-augmented generation over a local vector database (e.g., Qdrant)      |
| **General QA**          | Open-ended question answering and reasoning                                     |
| **Academic Governance** | Document-grounded QA over academic governance materials                         |

---

# OAuth Implementation

OAuth authentication can be integrated to enhance security and manage user access. For detailed setup instructions, refer to the [OAuth guide](docs/OAuth.md).

---

# Reference

If this repository is useful for your work, please cite the paper below:

```bibtex
@misc{
    title         = {Unleashing Artificial Cognition: Integrating Multiple AI Systems},
    author        = {Muntasir Adnan and Buddhi Gamage and Zhiwei Xu and Damith Herath and Carlos C. N. Kuhn},
    howpublished  = {Australasian Conference on Information Systems},
    year          = {2024}
}
```

---

# Contact

For technical support, please contact:
1. [Carlos Kuhn](mailto:carlos.kuhn@canberra.edu.au)
2. [Muntasir Adnan](mailto:adnan.adnan@canberra.edu.au)
3. [Zohaib Hammad](mailto:zohaib.hammad@canberra.edu.au)

For project support, please contact:
1. [Carlos C. N. Kuhn](mailto:carlos.noschangkuhn@canberra.edu.au)

---

# Contributing

We welcome any contributions from the community no matter if you are researchers, developers, and enthusiasts:

- **Report issues**: found a bug or have a feature request? Open an issue on our [GitHub page](https://github.com/TheOpenSI/CoSMIC/issues)
- **Submit pull requests**: please follow our [contribution guidelines](./CONTRIBUTING)
- **Make a donation**: support the project [here](https://payments.canberra.edu.au/Misc/tran?tran-type=OPENSI)

---

# License

This code is distributed under [the MIT License](./LICENSE)

If models from:
1. **Hugging Face (Mistral 7B, Gemma 7B, etc.)** are used, please also follow [Hugging Face's licence terms](https://huggingface.co/docs/hub/repositories-licenses)
2. **OpenAI API models (GPT-3.5-Turbo, GPT-4o, etc.)** are used, please also follow [OpenAI's licence terms](https://github.com/openai/openai-openapi/blob/master/LICENSE)

---

# Funding

This project is funded under the agreement with:
1. **ACT Government for Future Jobs Fund with Open Source Institute** (OpenSI-R01553)
2. **NetApp Technology Alliance Agreement** (OpenSI-R01657)
