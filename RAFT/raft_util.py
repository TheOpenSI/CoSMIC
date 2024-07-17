import os
from dotenv import load_dotenv
from typing import List

from llama_index.core import SimpleDirectoryReader
from llama_index.core.llms import ChatMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

def get_doc_chunk(file_path: str, separators: List[str] = ["\n\n", "\n"]) -> List[str]:
    docs = SimpleDirectoryReader(file_path).load_data()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        separators = separators,
        is_separator_regex = False
        )
    
    # we have a single document only
    chunks = text_splitter.split_text(docs[0].text)
    
    return [chunk.replace("\n", " ") for chunk in chunks]


def generate_qa_pair(llm:OpenAI, chunk:str, num_questions:int = 2) -> str:
    messages = [
        ChatMessage(
            role="system",
            content='''
            You are a synthetic question-answer pair generator. 
            Given a chunk of context about some topic(s), generate %s example questions a user could ask and would be answered using information from the chunk.
            Add chain-of-thought
            For example, if the given context was a Wikipedia paragraph about the United States, an example question could be 'How many states are in the United States?' and answer would be 50. 
            The questions should be able to be answered in a few words or less.
            Separate question and answer with "====="
            After each pair add "#####"'''
            % (num_questions),
        ),
        ChatMessage(role = "user", content = chunk),
    ]

    return str(llm.chat(messages))


def generate_cot_answer(llm:OpenAI, question:str, context:str) -> str:
    prompt = f"""
        Question: {question}\nContext: {context}\n
        Answer this question using the information given in the context above. Here is things to pay attention to:
        - First provide step-by-step reasoning on how to answer the question.
        - In the reasoning, if you need to copy paste some sentences from the context, include them in ##begin_quote## and ##end_quote##. This would mean that things outside of ##begin_quote## and ##end_quote## are not directly copy paste from the context.
        - Mention the context that's not relevant to the question.
        - End your response with final answer in the form <ANSWER>: $answer, the answer should be succinct.
    """
    messages = [
        ChatMessage(
            role="system",
            content="You are a helpful assistant who can provide an answer given a question and relevant context.",
        ),
        ChatMessage(role="user", content=prompt),
    ]
    return str(llm.chat(messages))


def flatten_list(xss: List[List[str]]):
    '''
    flattens a 2d array
    '''

    return [x for xs in xss for x in xs]


def process_qa_pair(qa_pair: str) -> List[str]:
    '''
    0, 2 will be questions and 1, 3 will be answers
    '''
    pairs = qa_pair.split("#####")[:-1]
    q_a = [pair.split("=====") for pair in pairs]
    return list(map(lambda s: s.replace("\n", " ").replace("assistant: ", " ").strip(), flatten_list(q_a)))
