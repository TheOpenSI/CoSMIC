echo "Starting Cosmic Chatbot"
echo "========================"
echo "Starting Ollama"
ollama serve > ollama.log 2>&1 &
echo "Starting Chatbot"
bash /app/scripts/chatbot/run_chatbot_docker.sh