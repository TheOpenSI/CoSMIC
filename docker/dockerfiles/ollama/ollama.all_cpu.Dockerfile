# All CPU Ollama setup (works on all OSes). For some reasons, 'latest' tag is
# using the Ollama 0.16.x version so until they corrected it, we have to pin to
# the current newest version.
FROM ollama/ollama:0.17.7 AS all_cpu_base

EXPOSE 11434/tcp
