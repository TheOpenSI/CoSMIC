# Cognitive AI

This repository contains the implementation of a Cognitive AI experiments

## Workflow

### 1. Progressive Move Dataset

- [x] **Collect Data**: LICHESS dataset from [Kaggle](https://www.kaggle.com/datasets/datasnaek/chess)
- [x] **Preprocess Data**: Create the progressive move dataset

### 2. Prompt Engineering Template

- [ ] **Design Templates**: Create chain-of-thought prompt template
- [ ] **Test Prompts**: Validate the prompts with OPENAI API

### 3. Generate Explanation

- [ ] **Xxplanation Dataset**: Use OPENAI API and the template to generate the dataset with move explanation

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

Create ```.env``` file and add the openai API token using the variable ```openai_api_key```