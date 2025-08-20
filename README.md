# OpenSI-CoSMIC - Cognitive System of Machine Intelligent Computing

[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/ACIS-2024-oliver.svg)](https://arxiv.org/abs/2408.04910)
[![python](https://img.shields.io/badge/Python-3.8-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![Media](https://img.shields.io/badge/Media-2024-purple.svg)](https://www.canberra.edu.au/about-uc/media/newsroom/2024/november/ucs-opensi-researchers-develop-framework-to-integrate-and-interpret-ai-tools)
[![DebianBadge](https://badges.debian.net/badges/debian/stable/docker/version.svg)](https://www.docker.com/)

This is the official implementation of the Open Source Institute-Cognitive System of Machine Intelligent Computing (OpenSI-CoSMIC) v1.0.0.

## Installation

### Pre-requirements

Before proceeding with the installation, ensure that the following tools are installed on your local machine:

1. **Docker**: Required for containerized environments. You can install it by following the [official Docker installation guide](https://docs.docker.com/get-docker/).

2. **Docker Compose**: Facilitates defining and running multi-container Docker applications. You can install it by following the [official Docker Compose installation guide](https://docs.docker.com/compose/install/).

### Option 1: Docker Installation (Quick Start)

The Docker installation provides the fastest way to get started with OpenSI-CoSMIC:

1. Download the `docker-compose.yaml` file from the official CoSMIC GitHub repository:

```bash
wget https://github.com/TheOpenSI/CoSMIC/raw/production/docker-compose.yaml https://github.com/TheOpenSI/CoSMIC/blob/dev/start.sh
```
2. **Important**: If you're running on a machine without an NVIDIA GPU or CUDA support, you need to modify the `docker-compose.yaml` file. Open the file and comment out the GPU resource allocation section:
```yaml
# Comment out these lines if you don't have an NVIDIA GPU
 deploy:
   resources:
     reservations:
       devices:
         - driver: nvidia
           count: all
           capabilities: [gpu]
```


3. Open the directory containing the downloaded files in a terminal and run the following command to start the services:

```bash
bash start.sh # bash start.sh --help for details
```

### Option 2: Clone and Set Up Repository (For Development)

1. Install Git on your local machine if it is not already installed. You can follow the [official Git installation guide](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

2. Clone the CoSMIC repository in your work directory:
```bash
# For users using SSH on GitHub
git clone git@github.com:TheOpenSI/CoSMIC.git

# For users using HTTPS
git clone https://github.com/TheOpenSI/CoSMIC.git
```

3. Clone the Open-WebUI repository in your work directory:
```bash
# For users using SSH on GitHub
git clone git@github.com:TheOpenSI/OpenWebUI-CoSMIC.git

# For users using HTTPS
git clone https://github.com/TheOpenSI/OpenWebUI-CoSMIC.git
```

**Note**: Ensure that both repositories are cloned into the same directory to maintain compatibility.

4. **Important**: If you're running on a machine without an NVIDIA GPU or CUDA support, you need to modify the `docker-compose.yaml` file. Open the file and comment out the GPU resource allocation section:
```yaml
# Comment out these lines if you don't have an NVIDIA GPU
 deploy:
   resources:
     reservations:
       devices:
         - driver: nvidia
           count: all
           capabilities: [gpu]
```

5. Navigate to the CoSMIC repository directory and start the services using Docker Compose:
```bash
cd CoSMIC
```

6. Now you can build from your local clone using the command below:
```bash
bash start.sh --docker_build
```
This will create an external volume required for [PyCapsule](https://github.com/TheOpenSI/PyCapsule) and run the `docker compose up` or `docker compose up --build` command. For details, run `bash start.sh --help`.

7. **Important**: During the first run, the system will automatically download the Llama3.1 model, which may take some time depending on your internet connection. You can monitor the progress by checking the Docker logs:
```bash
docker compose logs -f cosmic
```
Wait until you see the message: `cosmic | Model Llama3.1 is available in the Ollama container.` before attempting to use the application.

The application will initialize on port 8080. To access it, open a web browser and navigate to `http://localhost:8080`.

## OAuth Implementation
You can integrate OAuth authentication into this application to enhance security and manage user access. For detailed instructions on setting up OAuth, please refer to our [OAuth guide](OAuth.md).

## Postgres Implementation

By default, OpenSI-CoSMIC uses SQLite as its database. However, if you prefer to use Postgres for enhanced scalability and performance, you can configure it by following these steps:

1. Open the `.env` file in the root directory of the project and set the following variables:
  - `DATABASE_USER`: Specify the username for the Postgres database.
  - `DATABASE_PASSWORD`: Specify the password for the Postgres database.
  - `PGADMIN_USER`: Specify the username for PGAdmin. Ex. root@root.com
  - `PGADMIN_PASSWORD`: Specify the password for PGAdmin.

2. Once the `.env` file is configured, run the following command to start the services with Postgres:
```bash
docker compose -f docker-compose.postgres.yaml up -d
```

This will initialize the application with Postgres as the database backend.

**Note**: Configuring the `.env` file is mandatory for the Postgres setup to work correctly. Ensure all variables are properly set before starting the services.

## Framework
The system is configurated through [config.yaml](scripts/configs/config.yaml).
Currently, it has 5 base services, including

- [Chess-game next move predication and analyse](src/services/chess.py)
- [Vector database for text-based and document-base information update](src/services/vector_database.py)
- [Context retrieving through the vector database](src/services/rag.py) if applicable
- [PyCapsule (python code generation)](https://github.com/TheOpenSI/PyCapsule)
- [General question answering and reasoning](src/services/qa.py)

Each query will be parsed by [an LLM-based analyser](src/query_analyser/query_analyser.py) to select the most relevant service.

Upper-level chess-game services include

- [Puzzle next move prediction and analyse](src/modules/chess_qa_puzzle.py)
- [FEN generation given a sequence of moves](src/modules/chess_genfen.py)
- [Chain-of-Thought generation for next move prediction](src/modules/chess_gencot.py)

## Access Statistic
For Chatbot users, the user access information including the user ID, email, visit dates, average token length, and the number of queries are stored monthly.
- For docker users: /app/data/cosmic/statistic/[month]-[year].csv in the cosmic container.

## Reference
If this repository is useful for you, please cite the paper below.
```bibtex
@misc{
    title         = {Unleashing Artificial Cognition: Integrating Multiple AI Systems},
    author        = {Muntasir Adnan and Buddhi Gamage and Zhiwei Xu and Damith Herath and Carlos C. N. Kuhn},
    howpublished  = {Australasian Conference on Information Systems},
    year          = {2024}
}
```

## Contact
For technical supports, please contact [Carlos Kuhn](mailto:carlos.kuhn@canberra.edu.au), [Muntasir Adnan](mailto:adnan.adnan@canberra.edu.au) or [Zohaib Hammad](mailto:zohaib.hammad@canberra.edu.au).
For project supports, please contact [Carlos C. N. Kuhn](mailto:carlos.noschangkuhn@canberra.edu.au).

## Contributing

We welcome contributions from the community! Whether you’re a researcher, developer, or enthusiast, there are many ways to get involved:

 - Report Issues: Found a bug or have a feature request? Open an issue on our GitHub page.
 - Submit Pull Requests: Contribute code by submitting pull requests. Please follow [our contribution guidelines](CONTRIBUTING.md).
 - Make a Donation: Support our project by making a donation [here](https://payments.canberra.edu.au/Misc/tran?tran-type=OPENSI).

## License
This code is distributed under [the MIT license](LICENSE).
If Mistral 7B v0.1, Mistral 7B Instruct v0.1, Gemma 7B, or Gemma 7B It from Hugging Face is used, please also follow the license of Hugging Face;
if the API of GPT 3.5-Turbo or GPT 4-o from OpenAI is used, please also follow the licence of OpenAI.

## Funding
This project is funded under the agreement with the ACT Government for Future Jobs Fund with Open Source Institute (OpenSI)-R01553 and NetApp Technology Alliance Agreement with OpenSI-R01657.
