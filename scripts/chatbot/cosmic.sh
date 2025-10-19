echo "Starting Cosmic Chatbot"
echo "========================"
echo "Starting Ollama"
ollama serve > ollama.log 2>&1 &

# Ensure database migrations are applied before starting the API
echo "Running Alembic migrations..."
RETRIES=30
SLEEP=2
for i in $(seq 1 $RETRIES); do
	alembic upgrade head && SUCCESS=1 && break
	echo "Alembic attempt $i failed, retrying in ${SLEEP}s..."
	sleep $SLEEP
done
if [ -z "$SUCCESS" ]; then
	echo "Alembic migrations failed after $RETRIES attempts. Continuing anyway."
fi

# https://www.uvicorn.org/settings/#configuration-methods
uvicorn api:app --host 0.0.0.0 --port 3000