# Set Python environment.
FROM python:3.8.10

# Work directory in container.
WORKDIR /app

#port expose
EXPOSE 3000

# Copy all necessary files/folders to container folder.
COPY . . 
# COPY requirements.txt /app

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    curl \
    wget \
    vim \
    build-essential \
    docker.io \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Build environment.
RUN pip install -r requirements.txt

# Run main file.
# CMD ["python", "modules/docker/main_docker.py"]
# CMD ["bash", "run_chatbot.sh"]
CMD ["/bin/bash"]