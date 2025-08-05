#!/bin/bash
set -e

DOCKER_BUILD=false
KEEP_COSMIC=false

# Usage
show_usage(){
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --help                    Show this help message"
    echo "  --docker_build            Build the Docker image before starting the services"
    echo "  --cosmic_cli              Keep Cosmic interactive CLI running"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Default"
    echo "  $0 --docker_build                     # Build Docker image first"
    echo "  $0 --cosmic_cli                       # Keep Cosmic CLI running"
}

# Parse command line arguments
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
        --cosmic_cli)
            KEEP_COSMIC=true
            shift
            ;;
        *)
            echo "Error: Unknown option $1"
            show_usage
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Docker build: $DOCKER_BUILD"
echo "  Keep Cosmic CLI running: $KEEP_COSMIC"
echo ""

# Create Docker volume
echo "Creating Docker volume for Ollama..."
docker volume create shared_mount

# Start Docker Compose
echo "Starting Docker Compose services..."
if [[ "$DOCKER_BUILD" == true ]]; then
    docker compose up --build
elif [[ "$KEEP_COSMIC" == true ]]; then
    docker compose run --rm cosmic
else
    docker compose up
fi

exit 0