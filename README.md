# Cognitive AI

This repository contains the implementation of a Cognitive AI experiments

## Draft System Architecture
![System Architecture v-1.0](https://github.com/TheOpenSI/cognitive_AI_experiments/blob/master/draft_system_architecture_v1_0.png)

## Workflow

### 1. Progressive Move Dataset

- [x] **Collect Data**: LICHESS dataset from [Kaggle](https://www.kaggle.com/datasets/datasnaek/chess)
- [x] **Preprocess Data**: Create the progressive move dataset

### 2. Prompt Engineering Template

- [x] **Design Prompt Templates**: Create chain-of-thought prompt template
- [x] **Test Prompts**: Validate the prompts with OPENAI API

### 3. Generate Explanation

- [x] **Explanation Dataset**: Use OPENAI API and the template to generate the dataset with move explanation
- [x] **Validate Dataset**: Valdiate the generated dataset.

### 4. Select LLM

- [x] **Evaluate Options**: Review available open source Large Language Models with **System Prompt** compatibility
- [x] **Select Model**: Choose the most appropriate LLM. Selected model/s so far - **Mistral 7B instruct**, **Orca 2**  

### 5. Load LLM

- [x] **Load Model**: Quantization

### 6. Finetune

- [x] **Finetune Model:** LoRA, DoRA (Used LoRA for now)
- [x] **Evaluate Performance**: [OpenSI Finetuned Mistral](adnaan525/opensi_mistral_3tasks)

### 7. Retrieval-Augmented Generation (RAG)

- [x] **Setup RAG**: Implement the Retrieval-Augmented Generation process (learn new knowledge)
- [x] **Collect Documents**: Feed the model a comprehensive collection of chess books
- [x] **Vectore Store Sync**: Update vector store on the fly, **Inserting deprecatednew document**
- [x] **Test and Validate**: Conduct thorough testing to validate the integration and performance of the RAG system.

![Updating vector index on a new thread to gather new information](https://github.com/TheOpenSI/cognitive_AI_experiments/blob/master/RAG/RAG.png)

## Setup

### Docker
I used a docker image with all necessary libraries including jupyter-lab, torch, llama_index, langchain, bits_and_bytes, accelerate, transformers for local development.
```bash
docker pull ghost525/llm_finetune:latest
``` 

### Progressive Move Dataset
From Kaggle Lichess dataset, we created progessive move dataset to support a typical NLP training pipeline. Target dataset shape is ```(5000, 3)``` which had been splitted into 10 files, each with 500 rows.

### Explanation Data Generation [DEPRECATED]
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
## 2nd finetune
Our finetuned model has currently been introduced to the following tasks - 
- Move explanaton
- Provide next possible move [deprecated]
- Predict possible next winner  
The following will be added to the training dataset to diversify the range of tasks - 
- [x] Capture analysis
- [ ] FEN parsing and reasoning after FEN state
- [ ] Learn to use RAG context, may be useful for RAFT

## Cognition Framework test with LLama-Indexing Implementation
The congnition_test script requires an argument to specify the model, currently 3 models are supported.
```
$ python cognition_test.py --help
usage: cognition_test.py [-h] --model MODEL

optional arguments:
  -h, --help     show this help message and exit
  --model MODEL  [Required] mistral or gemma or llama
```
Example use
```
$ python cognition_test.py --model gemma
```
