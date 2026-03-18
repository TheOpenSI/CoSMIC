#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${SCRIPT_DIR}/../.."

image_name="open-webui"
container_name="open-webui"
host_port=3000
container_port=8080

# Dynamically get the host IP address
HOST_IP=$(hostname -i | awk '{print $1}')
echo "[INFO] Using host IP: $HOST_IP"

# Create a shared volume for the container
docker volume create volume_configs
docker volume create volume_uploads

ENV_CONTENT=""
if [[ -e ${ROOT}/.env ]]; then
    ENV_CONTENT=$(cat ${ROOT}/.env)
    echo "Read .env file content"
fi

# Build the image
docker build -t "$image_name" ${ROOT}/modules/chatbot
docker rm "$container_name" &>/dev/null || true

echo "[WARNING] For development purposes, the container is directly using the host network."
echo "[WARNING] This is not recommended for production deployments."

# Run the container
docker run -d \
    --network=host \
    --add-host=host.docker.internal:$HOST_IP \
    --name "$container_name" \
    --restart always \
    -v volume_configs:/app/backend/configs \
    -v volume_uploads:/app/backend/data/uploads \
    -e OPENAI_API_BASE_URL=http://host.docker.internal:9099 \
    -e OPENAI_API_KEY=0p3n-w3bu! \
    "$image_name"

docker image prune -f

# Run the pipeline script
cd ${ROOT}/modules/chatbot/pipelines
bash start_docker.sh
cd ${ROOT}