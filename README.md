# OpenSI-CoSMIC - Cognitive System of Machine Intelligent Computing

[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/ACIS-2024-oliver.svg)](https://arxiv.org/abs/2408.04910)
[![python](https://img.shields.io/badge/Python-3.14-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![Media](https://img.shields.io/badge/Media-2024-purple.svg)](https://www.canberra.edu.au/about-uc/media/newsroom/2024/november/ucs-opensi-researchers-develop-framework-to-integrate-and-interpret-ai-tools)

This is the official implementation of the Open Source Institute - Cognitive System of Machine Intelligent Computing (OpenSI-CoSMIC) v1.0.0, an innovative framework that integrates multiple AI systems into a unified cognitive computing platform.

---

## ⚠️ Important Notice

> [!WARNING]
> This branch represents a significant architectural shift. We are actively
> transitioning away from OpenWebUI to build our own UI platform. As a result,
> some components may be unstable during this migration phase. We recommend
> using this branch only if you're comfortable working with a system under
> active development.

> [!NOTE]
> The `backend/todo` directory contains the legacy setup configuration. While
> we're migrating to the new structure, you can safely ignore this directory
> unless you have specific needs from it.

---

## 📋 Quick Start

Before you begin, ensure you have Docker and Docker Compose installed on your system. These are required to run OpenSI-CoSMIC:

- **Docker**: [Installation Guide](https://docs.docker.com/get-docker/)
- **Docker Compose**: [Installation Guide](https://docs.docker.com/compose/install/)

Then, start by cloning the repository using your preferred method:

```bash
# Using HTTPS (recommended for most users)
git clone --single-branch -b bing-dev https://github.com/TheOpenSI/CoSMIC.git

# Using SSH (recommended if you have SSH keys configured)
git clone --single-branch -b bing-dev git@github.com:TheOpenSI/CoSMIC.git
```

Once cloned, navigate to the project directory:

```bash
cd CoSMIC
```

---

## 🚀 Getting Started

Depending on your use case, you have three main paths to explore. Whether you want to see the system in action, contribute to development, or understand the architecture, we've organized the documentation to guide you through each pathway.

### Option 1: Quick Demo

1. **Linux**
```bash
# Make sure you're in the root directory
chmod +x ./scripts/cosmic_demo.sh && ./scripts/cosmic_demo.sh
```

2. **macOS**
```zsh
# Make sure you're in the root directory
chmod +x ./scripts/cosmic_demo.zsh && ./scripts/cosmic_demo.zsh
```

3. **Windows**
```ps1
# Make sure you're in the root directory
.\scripts\CosmicDemo.ps1
```

#### **GPU Configuration**

OpenSI-CoSMIC includes flexible GPU support through Ollama. By default, the system uses a CPU-only Dockerfile that's guaranteed to work on all operating systems (Windows, macOS, Linux). However, if you have a GPU installed on a Linux or macOS machine and want to leverage it for improved performance, you can enable GPU acceleration.

To enable GPU support, open the `docker-compose.yaml` file and modify the Ollama service configuration. The file includes clear documentation on which Dockerfile to use based on your GPU type:

For NVIDIA GPUs on Linux or macOS, replace the default build configuration with:

```yaml
build:
  context: ./
  dockerfile: ./docker/dockerfiles/ollama/ollama.nvidia_cpu.Dockerfile
gpus: all
```

For AMD GPUs on Linux or macOS, use instead:

```yaml
build:
  context: ./
  dockerfile: ./docker/dockerfiles/ollama/ollama.amd_cpu.Dockerfile
devices:
  - "/dev/kfd:/dev/kfd"
  - "/dev/dri:/dev/dri"
```

It's important to note that GPU support is only reliably available on Linux and macOS. If you're using Windows, the Docker virtualization layer introduces complications with GPU passthrough, so we recommend sticking with the default CPU-only configuration. For more detailed information about Ollama's Docker GPU support, please refer to the [official Ollama documentation](https://docs.ollama.com/docker).

The default CPU-only Dockerfile is optimized for broad compatibility and will work reliably across all platforms and hardware configurations, so you don't need to make any changes unless you specifically want GPU acceleration.

#### **First Run Initialisation**
> [!NOTE]
> During the first run, the system will download and install a substantial
> number of AI/ML Python packages. Due to the comprehensive nature of these
> dependencies, the initial setup may take considerable time depending on your
> internet connection speed and system specifications. We recommend sitting
> back, grabbing a cup of coffee or tea, and letting the system complete the
> installation process without interruption. Everything will be ready when it
> finishes!

### Option 2: Development Setup

If you're contributing to the CoSMIC project or want to understand how the different components work together, this section guides you through setting up the development environment.

#### **Frontend Development**

The frontend is built as a separate repository to maintain clean separation of concerns. For detailed information about frontend development, including how to set up the development environment, run hot-reload servers, and contribute to the UI, start with the [Frontend Development Guide](https://github.com/TheOpenSI/CoSMIC_UI/blob/main/README.md)

#### **Backend Development**

The backend contains the core cognitive services and integrations. To understand the architecture, set up a development environment, and contribute new features, start with the [Backend Development Guide](backend/README.md)

## **Understanding the Framework Architecture**

The system's behavior is configured through the `config.yaml` file located at `scripts/configs/config.yaml`. This configuration drives the five core services:

1. **Chess Game Service** (`src/services/chess.py`): Provides next move prediction and detailed game analysis using advanced AI models
2. **Vector Database Service** (`src/services/vector_database.py`): Manages text-based and document-based information, enabling semantic search and retrieval
3. **Retrieval-Augmented Generation** (`src/services/rag.py`): Enhances question answering by retrieving relevant context from the vector database
4. **Code Generation Service** (PyCapsule): Generates Python code based on natural language descriptions, integrated from [TheOpenSI/PyCapsule](https://github.com/TheOpenSI/PyCapsule)
5. **Question Answering Service** (`src/services/qa.py`): Provides general-purpose reasoning and question answering capabilities

When a user submits a query, the [LLM-based Query Analyzer](src/query_analyser/query_analyser.py) evaluates the input and routes it to the most appropriate service. This intelligent routing ensures that specialized services handle their domain-specific tasks while general services provide fallback capability.

Advanced chess capabilities build on the core service:

- **Chess Puzzle Solving** (`src/modules/chess_qa_puzzle.py`): Specialized puzzle analysis and next move prediction
- **FEN Generation** (`src/modules/chess_genfen.py`): Converts move sequences into Forsyth-Edwards Notation for position analysis
- **Chain-of-Thought Reasoning** (`src/modules/chess_gencot.py`): Generates step-by-step reasoning for move recommendations

---

## 📚 Development Conventions & Resources

This project follows specific conventions for different components to maintain consistency and facilitate collaboration. Choose the section relevant to your work:

1. **[Git Workflow](doc/GIT.md)**: Guidelines for branching, committing, and pulling requests
2. **[Frontend Development](doc/REACT.md)**: React component patterns, styling, and component organization
3. **[Backend Development](doc/PYTHON.md)**: Python code style, service architecture, and API design
4. **[Database Schema](doc/POSTGRES.md)**: Database design patterns, migrations, and query optimization

---

## 🤝 Contributing

We welcome contributions from researchers, developers, and enthusiasts. There are multiple ways to get involved with the OpenSI-CoSMIC project:

**Report Issues**: Found a bug or have a feature suggestion? Open an issue on our [GitHub repository](https://github.com/TheOpenSI/CoSMIC/issues).

**Submit Code Contributions**: We accept pull requests from the community. Please review our [contribution guidelines](CONTRIBUTING.md) before submitting to ensure your contributions align with our standards.

**Support the Project**: Consider making a donation to support ongoing development at [our donations page](https://payments.canberra.edu.au/Misc/tran?tran-type=OPENSI).

---

## 📝 Citation

If you use OpenSI-CoSMIC in your research or project, please cite the following paper:

```bibtex
@misc{
    title         = {Unleashing Artificial Cognition: Integrating Multiple AI Systems},
    author        = {Muntasir Adnan and Buddhi Gamage and Zhiwei Xu and Damith Herath and Carlos C. N. Kuhn},
    howpublished  = {Australasian Conference on Information Systems},
    year          = {2024}
}
```

---

## 📧 Support & Contact

We're here to help with technical questions and project coordination:

**Technical Support**: For engineering questions, bug reports, or implementation issues:
- [Carlos Kuhn](mailto:carlos.kuhn@canberra.edu.au)
- [Muntasir Adnan](mailto:adnan.adnan@canberra.edu.au)
- [Zohaib Hammad](mailto:zohaib.hammad@canberra.edu.au)
- [Manile Srun](mailto:manile.srun@canberra.edu.au)
- [Bing Tran](mailto:binhsan1307@gmail.com)

**Project Management**: For project-level decisions and strategic inquiries:
- [Carlos C. N. Kuhn](mailto:carlos.noschangkuhn@canberra.edu.au)

---

## 📄 License

OpenSI-CoSMIC is distributed under the [MIT License](LICENSE). This permissive license allows broad use while maintaining attribution requirements.

**Important License Notes**: If you use any of the following models or services, you must also comply with their respective licenses:

- **Ollama Models**: OpenSI-CoSMIC uses Ollama for running large language models. The models available through Ollama (such as Llama3.1) have their own respective licenses. You must review and comply with the license of any specific model you download and use through Ollama. Visit the [Ollama models page](https://ollama.com/library) for detailed license information for each model.
- **Hugging Face Models**: If you use Mistral 7B v0.1, Mistral 7B Instruct v0.1, Gemma 7B, or Gemma 7B Instruct from Hugging Face, you must also follow the [Hugging Face Model License](https://huggingface.co/models)
- **OpenAI API**: If you integrate GPT 3.5-Turbo or GPT 4-o from OpenAI, you must also comply with [OpenAI's terms of service](https://openai.com/terms)

---

## 💰 Funding

This project receives funding through:

- **ACT Government Future Jobs Fund**: Partnership with Open Source Institute (OpenSI) under agreement R01553
- **NetApp Technology Alliance**: Partnership with NetApp under agreement R01657

These partnerships enable us to maintain the project, implement new features, and provide community support.
