# Set Python environment.
FROM python:3.11.12-slim

# Work directory in container.
WORKDIR /app

# Port
EXPOSE 3000

# Copy application files and requirements.txt explicitly.
COPY requirements.txt /app/requirements.txt
COPY . /app

# Build environment.
RUN pip install -r requirements.txt

# Entry point.
CMD ["bash", "/app/scripts/chatbot/cosmic.sh"]