# Set Python environment.
FROM python:3.8.10

# Work directory in container.
WORKDIR /app

# Port
EXPOSE 3000

# Copy all necessary files/folders to container folder.
COPY . . 

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    lshw \
    git \
    curl \
    wget \
    vim \
    build-essential \
    docker.io \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh -s /root/.ollama

# Build environment.
RUN pip install -r requirements.txt

# Fix cosmic permission
RUN chmod +x /app/scripts/chatbot/cosmic.sh

# Entry point.
CMD ["bash", "/app/scripts/chatbot/cosmic.sh"]