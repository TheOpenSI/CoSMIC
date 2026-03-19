#!/bin/bash
set -e

DOCKER_BUILD=false

show_usage(){
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --help                    Show this help message"
    echo "  --docker_build            Build the Docker image before starting the services"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Default"
    echo "  $0 --docker_build                     # Build Docker image first"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            show_usage
            exit 0
            ;;
        --docker_build)
            DOCKER_BUILD=true
            shift
            ;;
        *) echo "Error: Unknown option $1"
           show_usage
           exit 1
           ;;
    esac
done

echo "Configuration:"
echo "  Docker build: $DOCKER_BUILD"
echo ""

# Create volume
echo "Creating Docker volume for PyCapsule..."
docker volume create shared_mount

# Docker Compose
echo "Starting Docker Compose services..."
if [[ "$DOCKER_BUILD" == true ]]; then
    docker compose up --build
else
    docker compose up
fi

exit 0