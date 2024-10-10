# Copy the command below on demand and run on terminal.

# Delete docker containers with image name as opensi_docker.
docker rm $(docker ps -aq  --filter ancestor=opensi_docker)

# Delete docker images.
docker rmi -f $(docker images -aq)

# Build a docker image.
docker build -t opensi_docker .

# Run the image within a container, static.
docker run --gpus all -e query="Who are you?" opensi_docker

# Run without copy, but mount, dynamic.
docker run -v ${PWD}:/app -v ~/.cache/huggingface:/root/.cache/huggingface --gpus all -e query="What can you provide?" opensi_docker

# Run within interaction.
docker run -it -v ${PWD}:/app -v ~/.cache/huggingface:/root/.cache/huggingface --gpus all -e query="Who are you?" opensi_docker /bin/bash

# Check size.
docker ps --size