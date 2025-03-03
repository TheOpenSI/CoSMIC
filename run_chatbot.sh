#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
#echo "SCRIPT_DIR is: $SCRIPT_DIR"

image_name="open-webui"
container_name="open-webui"
host_port=3000
container_port=8080

# dynamically get the host IP address
HOST_IP=$(hostname -i | awk '{print $1}')
echo "[INFO] Using host IP: $HOST_IP"

# create a shared volume for the container
docker volume create ${container_name}-shared 2>/dev/null || true
# echo "Created Docker volume: ${container_name}-shared"

ENV_CONTENT=""
if [[ -e ${SCRIPT_DIR}/.env ]]; then
    ENV_CONTENT=$(cat ${SCRIPT_DIR}/.env)
    echo "Read .env file content"
fi

# build the image
docker build -t "$image_name" ${SCRIPT_DIR}/modules/chatbot
docker rm "$container_name" &>/dev/null || true

# sharing the host network so port forwarding is not needed
#    -p "$host_port":"$container_port" \
#    -p 9099:9099 \
echo "[WARNING] For development purposes, the container is directly using the host network."
echo "[WARNING] This is not recommended for production deployments."

# run the container
docker run -d \
    --network=host \
    --add-host=host.docker.internal:$HOST_IP \
    -v "${image_name}:/app/backend/data" \
    -v "${container_name}-shared:/app/backend/shared" \
    --name "$container_name" \
    --restart always \
    -e ENV_CONTENT="$ENV_CONTENT" \
    "$image_name"

docker image prune -f

# run the pipeline script
cd ${SCRIPT_DIR}/modules/chatbot/pipelines
bash start.sh
cd ${SCRIPT_DIR}
