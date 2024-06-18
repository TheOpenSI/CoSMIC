import threading
import time
import os
import torch
from dotenv import load_dotenv
from datetime import date

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

def update_vector_store(text_to_add : str, vector_index : llama_index.core.indices.vector_store.base.VectorStoreIndex, persist_dir="/home/s448780/workspace/cognitive_ai/RAG/persistent_store") -> None:
    print("[INFO] Updating vector store ..")
    document_to_add = Document.example()
    document_to_add.text = text_to_add
    vector_index.insert(document_to_add)
    time.sleep(10) # imitating large file
    print("[INFO] Vector Store updated")
    vector_index.storage_context.persist(persist_dir=persist_dir)

def loading(stop_event):
    animation = ["-", "/", "-", "\\"]
    idx = 0
    while not stop_event.is_set():
        print(animation[idx % len(animation)], end = "\r")
        idx += 1
        time.sleep(0.2)

def text_to_add(query):
    return f"{query.split('__update__store__')[-1].strip()} as of {date.today().strftime('%b %d, %Y')}"

def is_query(query, vector_index):
    if "__update__store__" in query:
        print("[Info] Keyword detected")

        # filtered text
        text = text_to_add(query)
    
        # control the ascii animation
        stop_event = threading.Event()

        # new thread to update vector store
        update_thread = threading.Thread(target = update_vector_store, args= (text, vector_index, ))
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

        return False
    
def auth():
    load_dotenv("/home/s448780/workspace/cognitive_ai/finetune/.env")
    hf_token = os.getenv("hf_token")
    login(hf_token)

def set_models(model_id = "Writer/camel-5b-hf"):

    # llm
    query_wrapper_prompt = PromptTemplate(
    "Always answer the question, even if the context isn't helpful."
    "Write a response that appropriately completes the request.\n\n"
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

def create_vector_store(input_dir = "/home/s448780/workspace/cognitive_ai/RAG/test_doc", persist_dir="/home/s448780/workspace/cognitive_ai/RAG/persistent_store"):
    documents = SimpleDirectoryReader(input_dir).load_data()
    # checking added documents
    print_book_names(documents)
    # indexing
    vector_index = VectorStoreIndex.from_documents(documents)
    vector_index.storage_context.persist(persist_dir=persist_dir)
    return vector_index
    

if __name__ == "__main__":
    # authenticate huggingface
    auth()

    set_models()
    vector_index = create_vector_store()

    engine = vector_index.as_query_engine()
    query = ""
    while query != "exit":
        query_flag = True
        query = input("[INFO] Please type in your query or update keyword '__update__store__' to update vector store or 'exit' to exit.\n[QUERY] ")
        flag = is_query(query, vector_index)
        query_flag =  False if flag == False or query == "exit" else True # this needs fixing
        if query_flag == False:
            storage_context = StorageContext.from_defaults(persist_dir="/home/s448780/workspace/cognitive_ai/RAG/persistent_store")
            vector_index = load_index_from_storage(storage_context)
            engine = vector_index.as_query_engine()
        if query_flag:
            print(f"[RESPONSE] {engine.query(query)}")