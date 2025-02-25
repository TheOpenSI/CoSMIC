# Set Python environment.
FROM python:3.8.10

# Work directory in container.
WORKDIR /app

#port expose
EXPOSE 3000

# Copy all necessary files/folders to container folder.
COPY . /app  
# COPY requirements.txt /app


# Build environment.
RUN pip install -r requirements.txt

# Run main file.
# CMD ["python", "modules/docker/main_docker.py"]
# CMD ["bash", "run_chatbot.sh"]
CMD "bin bash"

