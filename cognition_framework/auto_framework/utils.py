import threading
import time
import os
import torch
from dotenv import load_dotenv
from datetime import date

from engine_service import get_best_move

import llama_index
from llama_index.core import SimpleDirectoryReader # to load docs
from llama_index.core import Document, StorageContext, load_index_from_storage

from llama_index.core.prompts import PromptTemplate # generate prompt template with role
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import ChatPromptTemplate

from llama_index.core.response.notebook_utils import display_response # formatting


from llama_index.llms.huggingface import HuggingFaceLLM # llm

from llama_index.embeddings.huggingface import HuggingFaceEmbedding # embedding

from llama_index.core import Settings # pass llm and embedding

from llama_index.core import VectorStoreIndex # store vector store

from huggingface_hub import login

import warnings
warnings.filterwarnings("ignore")

# uses mock sleep
def update_vector_store(text_to_add : str, vector_index : VectorStoreIndex, persist_dir : str) -> None:
    print("[INFO] Updating vector store ..")
    document_to_add = Document.example()
    document_to_add.text = text_to_add
    vector_index.insert(document_to_add)
    print("[INFO] Vector Store updated")
    vector_index.storage_context.persist(persist_dir=persist_dir)
    time.sleep(5)
    print("[INFO] Persistent store updated")

def loading(stop_event):
    animation = ["-", "/", "-", "\\"]
    idx = 0
    while not stop_event.is_set():
        print(animation[idx % len(animation)], end = "\r")
        idx += 1
        time.sleep(0.4)

def text_to_add(query):
    return f"{query.split('__update__store__')[-1].strip()} as of {date.today().strftime('%b %d, %Y')}"

def get_moves(query):
    return query.split('__next__move__')[-1].strip()
    
def filter_query(query:str, vector_store:VectorStoreIndex, persist_dir:str):
    is_query = False

    # terminate
    if query == "exit":
        return False
    
    # update_store -- next move -- query
    if "__update__store__" in query:
        # update vector store
        print("[INFO] Update command detected")
        add_info = text_to_add(query)
        # thread controller
        stop_event = threading.Event()
        # update thread
        update_thread = threading.Thread(target = update_vector_store, args= (add_info, vector_store, persist_dir, ))
        update_thread.start()
        # ascii thread
        ascii_thread = threading.Thread(target=loading, args = (stop_event, ))
        ascii_thread.start()
        # waiting for update_thread to finish indexing
        update_thread.join()
        # stop ascii animation
        stop_event.set()
        ascii_thread.join()
        print("[INFO] Continuing on the main thread")

    elif "__next__move__" in query:
        # call engine service
        print("[INFO] Next Move command detected. Calling chess engine service")
        query_moves = get_moves(query)
        stop_event = threading.Event()
        chess_engine_thred = threading.Thread(target=get_best_move, args=(query_moves, ))
        chess_engine_thred.start()
        ascii_thread = threading.Thread(target=loading, args = (stop_event, ))
        ascii_thread.start()
        chess_engine_thred.join()
        stop_event.set()
        ascii_thread.join()
        print("[INFO] Continuing on the main thread")

        get_best_move(query_moves)
    else:
        is_query = True
    return is_query

    
def auth():
    load_dotenv("/home/s448780/workspace/cognitive_ai/finetune/.env")
    hf_token = os.getenv("hf_token")
    login(hf_token, add_to_git_credential=False)

def set_models(model_id):

    # llm
    query_wrapper_prompt = PromptTemplate(
    "Always answer the question, even if the context isn't helpful."
    "Write a response that appropriately completes the request, do not write any explanation, only answer.\n\n"
    "### Instruction:\n Using the most updated information, {query_str}\n\n### Response:")
    llm = HuggingFaceLLM(
    context_window=2048,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0, "do_sample": False},
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name=model_id,
    model_name=model_id,
    device_map="auto",
    tokenizer_kwargs={"max_length": 2048},
    model_kwargs={"torch_dtype": torch.float16, "load_in_8bit" : True})

    # embedding
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

    Settings.llm = llm
    Settings.embed_model = embed_model

def print_book_names(documents):
    books = set()
    for book in documents:
        books.add(book.metadata["file_name"])
    for book in books:
        print(f"[INFO] {book} added")

def create_vector_store(input_dir, persist_dir):
    documents = SimpleDirectoryReader(input_dir).load_data()
    # checking added documents
    print_book_names(documents)
    # indexing
    vector_index = VectorStoreIndex.from_documents(documents)
    vector_index.storage_context.persist(persist_dir=persist_dir)
    return vector_index

def reload_from_persist(persist_dir):
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    vector_index = load_index_from_storage(storage_context)
    return vector_index.as_query_engine()
