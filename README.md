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
- [x] **Vectore Store Sync**: Update vector store on the fly, **Inserting new document**
- [x] **Test and Validate**: Conduct thorough testing to validate the integration and performance of the RAG system.

![Updating vector index on a new thread to gather new information](https://github.com/TheOpenSI/cognitive_AI_experiments/blob/RAG/RAG/RAG.png)

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
## Cognition Framework test
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

## OpenSI Evaluation System
The prototype of the evaluation system is showcased by running
```
cd cognition_framework;
python opensi_eval_system.py
```
It will return the success ratio of testing samplings given in
```
cognition_framework/tests/test.csv
```
Questions with keyword "skip" will be ignored during evaluation.

**Functions supported**
- General question answering
- Question answering with context parsed from documents
- Next move prediction for chess games
- Solution prediction for chess puzzles

**Engines of the evaluation system**
- Large language model engine in [`llm.py`](./cognition_framework/engines/llm_engine/llm.py)
- Chess engine (with API of Stockfish) in [`chess.py`](./cognition_framework/engines/chess_engine/chess.py)

**API demo**
```
# Filter out low-confidence retrieved context if context document/text is provided, default retrieve_score_threshold=0.0
qa_system = OpenSIEvalSystem(retrieve_score_threshold=0.7)

# Provide a document path, .pdf only
qa_system.add_documents([document path])

# Provide a document directory containing multiple .pdf files
qa_system.add_document_directory([document directory])

# Set up a question
query = "What is the capital of Australia?"

# Call the evaluation system entry with up to topk tokens from the context document(s) provided
answer = qa_system(query, topk=5)

# Print out the testing sample with answer
print(set_color('info', f"Query: {query}.\n==> Answer: {answer}."))

# Release the system
qa_system.quit()
```
