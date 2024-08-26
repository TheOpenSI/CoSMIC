# -------------------------------------------------------------------------------------------------------------
# File: opensi_ai_system.py
# Project: OpenSI AI System
# Contributors:
#     Danny Xu <danny.xu@canberra.edu.au>
#     Muntasir Adnan <adnan.adnan@canberra.edu.au>
# 
# Copyright (c) 2024 Open Source Institute
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without
# limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so, subject to the following
# conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial
# portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
# LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# -------------------------------------------------------------------------------------------------------------

import os, sys, yaml

sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/..")

from src.llms import llm as llm_instances
from src.llms.llm import get_instance
from src.maps import LLM_INSTANCE_DICT
from src.services.vector_database import VectorDatabase
from src.services.qa import QABase
from src.services.rag import RAGBase
from modules.chess.chess_qa_puzzle import PuzzleAnalyse
from modules.chess.chess_qa_quality import QualityEval
from modules.chess.chess_genfen import FENGenerator
from modules.chess.chess_gencot import CotGenerator
from modules.code_generation.code_generation import CodeGenerator
from utils.log_tool import set_color
from box import Box

# =============================================================================================================

class OpenSIAISystem:
    def __init__(
        self,
        llm_name: str="",
        config_path: str="configs/config.yaml"
    ):
        """ Construct OpenSI AI System instance. It contains LLM and services including vector database
        and RAG, where RAG includes context retriever and vector database update.
        Chess services are induced in PuzzleAnalyse and QualityEval, called on demand, not as global instance.

        Args:
            llm_name (str): LLM name, check LLM_MODEL_DICT in src/maps.py, if it is empty, the entry
            is self.config.llm_name.
            config_path (str): path of configuration file.
        """
        # Check if required config file exists.
        if not os.path.exists(config_path):
            print(set_color("error", f"Config file {config_path} not exist."))
            sys.exit()

        # Load yaml file to get the config.
        self.config = Box.from_yaml(filename=config_path, Loader=yaml.FullLoader)

        # If llm_name is not specified, read it from the config file.
        if llm_name == "":
            llm_name = self.config.llm_name

        # Check if llm_name is supported.
        if llm_name not in LLM_INSTANCE_DICT.keys():
            print(set_color("error", f"Unsupported LLM: {llm_name}."))
            sys.exit()

        # Build LLM instance from class defined in .py
        self.llm = get_instance(llm_instances, LLM_INSTANCE_DICT[llm_name])(seed=self.config.seed)

        # Create vector database service which will be included in RAG for retrieve and information updates.
        vector_database = VectorDatabase()

        # Add a directory of documents.
        if os.path.exists(self.config.doc_directory):
            vector_database.add_document_directory(self.config.doc_directory)

        # Add documents.
        if self.config.document_path != "" or len(self.config.document_path) > 0:
            vector_database.add_documents(self.config.document_path)

        # Base RAG service with vector_database, the database can be changed using
        # self.rag.set_vector_database().
        self.rag = RAGBase(
            vector_database=vector_database,
            retrieve_score_threshold=self.config.rag.retrieve_score_threshold,
            topk=self.config.rag.topk
        )

        # QA module to handle basic types of questions, such __next__move__, __update__store__, and
        # general questions.
        self.qa = QABase(self.llm, self.rag)

    def quit(self):
        """ Release memory of LLM and vector embedding model in vector_database.
        """
        self.llm.quit()
        self.rag.vector_database.quit()

    def __call__(
        self,
        question: str,
        context: str="",
        log_file: str=None
    ):
        """Execute QA

        Args:
            question (str): a question or a .csv containing multiple questions.
            context (str, optional): context for this question. Defaults to "".
            log_file (str, optional): whether to print the result in a .txt file. Defaults to None.

        Returns:
            response (str): (truncated) response.
            raw_response (str): raw response from LLM without truncations.
            retrieve_score (float): context retrieve score if is_rag=True.
        """
        # Set initial output to return.
        response = None
        raw_response = None
        retrieve_score = -1

        # Chat-mode LLM do not need example in the system prompt.
        if self.llm.llm_name in ["mistral-7b-instruct-v0.1", "gemma-7b-it"]:
            use_example = False
        else:
            use_example = True

        # Check if the question is a string or a .csv file containing multiple sub-questions.
        if question.find(".csv") > -1:
            # Batch process for puzzles (move prediction and analysis) and 4 other quality evaluations.
            if question.find("puzzle") > -1:
                # Do not truncate from the LLM base class but instead using
                # PuzzleAnalyse.is_truncate_response=True.
                self.llm.set_truncate_response(False)

                # Truncation external to the system_prompter does not require keywords, no example
                # in the system prompt is required.
                self.llm.system_prompter.set_use_example(False)

                # Build PuzzleAnalyse QA service.
                puzzle_analyser = PuzzleAnalyse(
                    llm=self.llm,
                    rag=self.rag,
                    is_rag=False,
                    log_file=log_file,
                    is_truncate_response=True,
                    next_move_predict_backend="stockfish",
                )

                # Batch process the question file.
                average_score = puzzle_analyser.batch_process(question)

                # Return how many questions get correct best move prediction
                # using Stockfish or GPT API.
                response = f"Success rate of {question} is {average_score:.2f}."
            elif question.find("attention") > -1 \
                or question.find("memory") > -1 \
                or question.find("perception") > -1:

                # Manually switch on and off RAG for specific questions.
                if question.find("attention") > -1 \
                    or question.find("memory_update") > -1:
                    is_rag = True
                else:
                    is_rag = False

                # Truncate the response according to the system_prompt format.
                self.llm.set_truncate_response(True)

                # Use example's keywords for truncation for non-chat LLM.
                self.llm.system_prompter.set_use_example(use_example)

                # Build quality evaluation service.
                quality_evaluator = QualityEval(
                    llm=self.llm,
                    rag=self.rag,
                    is_rag=is_rag,
                    log_file=log_file
                )

                # Batch process the question file.
                average_score = quality_evaluator.batch_process(question)

                # Return how many questions get correct answer from LLM compared to GT answer.
                response = f"Success rate of {question} is {average_score:.2f}."
            elif question.find("checkmate_moves") > -1:
                # Generate FEN with moves
                fen_generator = FENGenerator(log_file=log_file)

                # Batch process the question file.
                fen_generator.batch_process(question)
            elif question.find("finetune_dataset") > -1:
                # Generate CoT analysis for finetune dataset.
                cot_generator = CotGenerator(
                    log_file=log_file,
                    is_truncate_response=True
                )

                # Batch process the question file.
                cot_generator.batch_process(question)
            elif question.find("code_generation") > -1:
                # Build code_generator in modules/code_generation/code_generation.py
                code_generator = CodeGenerator(
                    log_file=log_file,
                )

                # Batch process the question file.
                response = code_generator(question)
        else:
            # General question needs truncation according the system prompt to avoid hallucination.
            self.llm.set_truncate_response(True)

            # Truncation needs keywords from the example of system prompt.
            self.llm.system_prompter.set_use_example(True)

            # Process each question.
            response, raw_response, retrieve_score = self.qa(
                question,
                context=context,
                is_rag=True
            )

        # Return answers with and without truncation, and retrieve score if applicable otherwise -1.
        return response, raw_response, retrieve_score