#!/bin/bash

cd "$(dirname "$0")"
SCRIPT_DIR="$(pwd)"

image_name="open-webui"
container_name="open-webui"
host_port=3000
container_port=8080

if [[ ! -e .env ]]; then
    touch .env
fi

docker build -t "${image_name}" "${SCRIPT_DIR}/modules/chatbot"
docker stop ${container_name}
docker rm ${container_name}

mkdir -p ${SCRIPT_DIR}/data/cosmic/shared
mkdir -p ${SCRIPT_DIR}/data/cosmic/backend/uploads
mkdir -p ${SCRIPT_DIR}/data/cosmic/vector_db_cosmic
mkdir -p ${SCRIPT_DIR}/data/cosmic/statistic
mkdir -p ${SCRIPT_DIR}/data/cosmic/data

docker run -d \
    -p "$host_port":"$container_port" \
    --add-host=host.docker.internal:host-gateway \
    --name "$container_name" \
    --restart always \
    -v ${SCRIPT_DIR}/.env:/app/backend/.env \
    -v ${SCRIPT_DIR}/scripts/configs:/app/backend/configs \
    -v ${SCRIPT_DIR}/data/cosmic/shared:/app/backend/data/shared \
    -v ${SCRIPT_DIR}/data/cosmic/backend/uploads:/app/backend/data/uploads \
    -v ${SCRIPT_DIR}/data/cosmic/data:/app/backend/data \
    -e OPENAI_API_BASE_URL=http://host.docker.internal:9099 \
    -e OPENAI_API_KEY=0p3n-w3bu! \
    -e DATABASE_URL="postgres://[your_postgred_path]" \
    "$image_name"