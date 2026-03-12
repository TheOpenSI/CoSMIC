# AMD GPU Ollama setup (works on Linux/MacOS only). For some reasons, 'latest'
# tag is using the Ollama 0.16.x version so until they corrected it, we have to
# pin to the current newest version.
FROM ollama/ollama:0.17.7-rocm AS amd_gpu_base
