# -------------------------------------------------------------------------------------------------------------
# File: maps.py
# Project: OpenSI AI System
# Contributors:
#     Danny Xu <danny.xu@canberra.edu.au>
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

LLM_MODEL_DICT = {
    "mistral-7b-v0.1": "mistralai/Mistral-7B-v0.1",
    "mistral-7b-instruct-v0.1": "mistralai/Mistral-7B-Instruct-v0.1",
    "gemma-7b": "google/gemma-7b",
    "gemma-7b-it": "google/gemma-7b-it",
    "mistral-7b-finetuned": "OpenSI/cognitive_AI_finetune_3",
    "gpt-3.5-turbo": "gpt-3.5-turbo",
    "gpt-4o": "gpt-4o"
}

# =============================================================================================================

LLM_INSTANCE_DICT = {
    "mistral-7b-v0.1": "Mistral7bv01",
    "mistral-7b-instruct-v0.1": "Mistral7bInstructv01",
    "gemma-7b": "Gemma7b",
    "gemma-7b-it": "Gemma7bIt",
    "mistral-7b-finetuned": "MistralFinetuned",
    "gpt-3.5-turbo": "GPT35Turbo",
    "gpt-4o": "GPT4o"
}