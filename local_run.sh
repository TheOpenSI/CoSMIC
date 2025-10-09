#!/bin/bash

cd "$(dirname "$0")"
SCRIPT_DIR="$(pwd)"

docker image prune -f

cd ${SCRIPT_DIR}/modules/chatbot/pipelines
bash start.sh
cd ${SCRIPT_DIR}