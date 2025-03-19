# -------------------------------------------------------------------------------------------------------------
# File: code_generation.py
# Project: Open Source Institute-Cognitive System of Machine Intelligent Computing (OpenSI-CoSMIC)
# Contributors:
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

import os, sys
import pandas as pd
import requests
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../..")

from typing import List
from src.services.qa import QABase
from utils.log_tool import set_color

# =============================================================================================================

class CodeGenerator(QABase):
    def __init__(self,
                 service_container_name: str = "loaclhost",
                 model_name: str = "qwen2.5-coder",
                 **kwargs):
        """
        Code generation module.
        Sends http requests at port 8780 to the PyCapsule service.
        
        Args:
            service_container_name (str): Name of the container, set to localhost for development.
                Set this to the container name for docker network.
            model_name (str): Model name.
        """
        super().__init__(**kwargs)
        self.model_name = model_name
        self.PORT = 8780
        self.service_url = f"http://{service_container_name}:{self.PORT}"
        self._check_service_status()
        
        
    def _check_service_status(self):
        """
        Check if the PyCapsule service is running.
        """
        try:
            response: requests.Response = requests.post(f"{self.service_url}/health")
            if response.status_code != 200:
                raise Exception(f"PyCapsule service is not running. Status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to connect to PyCapsule service: {str(e)}")
        
    
    def __call__(self, 
                 query:str, 
                 timeout: int = 30) -> tuple[str, str]:
        """
        Send the user query to the PyCapsule service.

        Args:
            query (str): User query.
            timeout (int): Timeout for the request. Defaults to 30 seconds.
        """
        try:
            pycaspsule_response: requests.Response = requests.post(f"{self.service_url}/query", 
                                                                      json={"query": query},
                                                                      timeout=timeout)
            
            if pycaspsule_response.status_code != 200:
                raise Exception((f"PyCapsule service returned error. "
                                 f"Status code: {pycaspsule_response.status_code}"))
            
            # Extract respone.
            full_response: str = pycaspsule_response.json().get("response", "")
            error: str = pycaspsule_response.json().get("error", "")
            code: str = pycaspsule_response.json().get("code", "")
            status: str = pycaspsule_response.json().get("status", "")

            raw_response, response = ((full_response, code)
                                      if status == "success"
                                      else (full_response, code + "\n\n" + error))
            
            return raw_response, response
        
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to connect to PyCapsule service: {str(e)}")