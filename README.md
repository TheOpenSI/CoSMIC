# Cognitive AI

This repository contains the implementation of a Cognitive AI experiments

## Workflow

### 1. Progressive Move Dataset

- [x] **Collect Data**: LICHESS dataset from [Kaggle](https://www.kaggle.com/datasets/datasnaek/chess)
- [x] **Preprocess Data**: Create the progressive move dataset

### 2. Prompt Engineering Template

- [x] **Design Prompt Templates**: Create chain-of-thought prompt template
- [x] **Test Prompts**: Validate the prompts with OPENAI API

### 3. Generate Explanation

- [ ] **Explanation Dataset**: Use OPENAI API and the template to generate the dataset with move explanation

### 4. Select LLM

- [ ] **Evaluate Options**: Review available open source Large Language Models with **System Prompt** compatibility
- [ ] **Select Model**: Choose the most appropriate LLM

### 5. Load LLM

- [ ] **Load Model**: LoRA, DoRA, Quantization

### 6. Finetune

- [ ] **Finetune Model**
- [ ] **Evaluate Performance**

### 7. Retrieval-Augmented Generation (RAG)

- [ ] **Setup RAG**: Implement the Retrieval-Augmented Generation process to enhance the model's responses with relevant information retrieval.
- [ ] **Collect Chess Documents**: Feed the model a comprehensive collection of chess books and other form of documents
- [ ] **Test and Validate**: Conduct thorough testing to validate the integration and performance of the RAG system.

## Setup

### Docker
I used a docker image with all necessary libraries including jupyter-lab, torch, llama_index, langchain, bits_and_bytes, accelerate, transformers for local development.
```bash
docker pull ghost525/llm_finetune:latest
``` 

### Progressive Move Dataset
From Kaggle Lichess dataset, we created progessive move dataset to support a typical NLP training pipeline. Target dataset shape is ```(5000, 3)``` which had been splitted into 10 files, each with 500 rows.

### Explanation Data Generation
The ```generate_data.py``` will generate explanation data and save in a CSV file automatically. The repo contains a codespace with all dependencies preinstalled on the master branch. For local development please follow the follwing procedures -   
- Resolve dependencies
    ```bash
    pip install -r requirements.txt
    ```
- OpenAI API key - We can pass our API key using a **txt file, .env file or manually pasting it as an argument in the terminal.** Example use cases-  
    Follwing options have been added to the ```generate_data.py```
    ```bash
    $ python generate_data.py --help
    usage: generate_data.py [-h] --file FILE [-i I] [-mI]

    options:
    -h, --help  show this help message and exit
    --file      [Required] path to the input data cluster(csv file), range 0-9(inclusive)
    -i          use a .txt file to pass token
    -mI         manually enter your token
    ```
    example use cases:
    ```
    python generate_data.py -i txt_file_path --file 0
    python generate_data.py -mI --file 0 # you will be prompted to paste your api key
    python generate_data.py --file 0 # default, use .env file, create .env file and add the openai API token using the variable "openai_api_key"
    ```
    example output:
    ```
    $ python generate_data.py -mI --file 9
    [Info] User will be requested for token
    Please enter your OpenAI token.
    __________paste_token__________
    [Info] Generating explanation for row 0
    [Info] Generating explanation for row 1
    [Info] Generating explanation for row 2
    [Info] Generating explanation for row 3
    [Info] Generating explanation for row 4
    [Info] CSV generated
    ```