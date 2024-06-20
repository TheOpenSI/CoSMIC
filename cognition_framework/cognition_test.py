import warnings
import logging
# logging set to warning
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("bitsandbytes").setLevel(logging.WARNING)
logging.getLogger().setLevel(logging.WARNING)
# suppress all warnings
warnings.filterwarnings("ignore")

import os
import argparse
import pandas

from llama_index.core import VectorStoreIndex # store vector store

from utils import auth, set_models, create_vector_store, filter_query, reload_from_persist

PERSIST = "vector_store"
target_models = {
    "mistral" : "mistralai/Mistral-7B-v0.1",
    "gemma" : "google/gemma-7b",
    "llama" : "meta-llama/Meta-Llama-3-8B-Instruct"
}

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="[Required] mistral or gemma or llama")

args = parser.parse_args()
target_repo = target_models[args.model]

print(f"[INFO] Selected model {target_repo}")

auth()
# loading llm and embedding
set_models(model_id=target_repo) # embedding set to "BAAI/bge-base-en-v1.5"
# vector store
vector_store = create_vector_store(input_dir="doc", persist_dir=PERSIST)

while True:
# query
    query = input("[INPUT] Please type in your query. Keywords : '__update__store__', '__next__move__'.\n[QUERY] ")
    if(query == "exit"):
        break
    flag = filter_query(query, vector_store, PERSIST)
    query_engine = reload_from_persist(PERSIST)
    if flag:
        print(f"[RESPONSE] {query_engine.query(query)}")