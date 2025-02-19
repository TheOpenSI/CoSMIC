# Set Python environment.
FROM python:3.8.10

# Work directory in container.
WORKDIR /app

# Copy all necessary files/folders to container folder.
COPY . /app

# Install Ollama.
RUN apt-get update
RUN apt-get -y install lshw
RUN curl -fsSL https://ollama.com/install.sh | sh -s /root/.ollama
RUN ollama --version

# Build environment.
RUN pip install -r requirements.txt

# Run main file.
CMD ["bash", "scripts/demo/run_demo.sh"]