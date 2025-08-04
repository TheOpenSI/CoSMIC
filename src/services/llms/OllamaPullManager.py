# -------------------------------------------------------------------------------------------------------------
# File: OllamaPullManager.py
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
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../../..")

import ollama
import threading
import time
from typing import List

class OllamaPullManager:
    def __init__(self, 
                 model_name: str, 
                 mode: str = "auto", 
                 interventions: list = [80, 90, 95], 
                 min_speed_kbps: float = 200.0, 
                 max_retries: int = 5,
                 fall_back_interval: int = 60,
                 ollama_client: ollama.Client = None):
        """
        OllamaPullManager to mange fail-safe model pulling

        Args:
            model_name (str): Name of the model to pull
            mode (str, optional): auto | speed | stochastic. Defaults to "auto".
            interventions (list, optional): List of percentages to intervene at. Defaults to [85].
            min_speed_kbps (float, optional): Minimum internet speed in kbps. Defaults to 200.0.
            max_retries (int, optional): Maximum retries for pulling the model. Defaults to 5.
            fall_back_interval (int, optional): Fallback interval in seconds. Defaults to 60.
        """
        self.model_name = model_name
        self.mode = mode  # "auto", "speed", "stochastic"
        self.interventions = interventions
        self.min_speed_kbps = min_speed_kbps
        self.max_retries = max_retries
        self.fall_back_interval = fall_back_interval
        if ollama_client is None:
            self.ollama_client = ollama.Client()
        else:
            self.ollama_client = ollama_client
        
        # Limit interventions to 5 (for Stochastic mode)
        if len(self.interventions) > 5:
            self.interventions = self.interventions[:5]
        
        # Thread management
        self._is_pulling = False
        self._should_stop = False
        self._download_thread = None
        
        # Progress tracking
        self._current_percentage = 0
        self._last_completed = 0
        self._download_completed = False
        
        # Intervention tracking
        self._intervention_count = 0
        self._completed_interventions = []
    
    
    def _reset(self):
        """
        Reset all after a pull.
        """
        self._is_pulling = False
        self._should_stop = False
        self._download_thread = None
        self._current_percentage = 0
        self._last_completed = 0
        self._download_completed = False
        self._intervention_count = 0
        self._completed_interventions.clear()
        

    def _get_available_models(self) -> List[str]:
        """
        Get the model names from the ollama.list() output.

        Returns:
            List[str]: List of available model names.
        """
        available_models = self.ollama_client.list()
        return [model.get("model") for model in available_models.get("models", [])]
    
    
    def _is_model_available(self):
        """
        Check if the model is avaiable on the server.
        """
        available_models = self._get_available_models()
        return True \
               if self.model_name in available_models \
               else False
        
        
    def pull_model(self) -> None:
        """
        Pull the model with the specified mode
        """
        flag = self._is_model_available()
        if flag:
            print(f"Model {self.model_name} is available on the server.")
            return
        
        print(f"Pulling model {self.model_name} in {self.mode} mode...")
        
        attempt = 0
        while attempt < self.max_retries:
            try:
                if self.mode == "stochastic":
                    success = self._pull_with_stochastic()
                elif self.mode == "speed":
                    success = self._pull_with_speed_monitoring()
                else:  # auto mode
                    success = self._pull_with_auto()
                
                if success:
                    self._reset()
                    print(f"\nModel {self.model_name} pulled successfully!")
                    return
                
            except Exception as e:
                print(f"\nError in attempt {attempt + 1}: {e}")
            
            attempt += 1
            if attempt < self.max_retries:
                print(f"Retrying... (Attempt {attempt + 1}/{self.max_retries})")
                time.sleep(2)
        
        self._reset()
        print(f"\nFailed to pull model after {self.max_retries} attempts.")
    
    
    def _pull_with_stochastic(self):
        """
        Pull with stochastic interventions
        """
        for intervention_idx, target_percentage in enumerate(self.interventions):
            if self._intervention_count >= 5:  # Max 5 interventions
                break
                
            print(f"Starting download (Intervention {intervention_idx + 1}/{len(self.interventions)}"
                       f" at {target_percentage}%)...")
            
            self._start_download()
            
            # Monitor progress until target percentage
            while self._is_pulling:
                if self._current_percentage >= target_percentage:
                    self._intervention_count += 1
                    self._completed_interventions.append({
                        'intervention': intervention_idx + 1,
                        'target_percentage': target_percentage,
                        'actual_percentage': self._current_percentage,
                        'status': 'completed'
                    })
                    
                    print(f"\nIntervention {intervention_idx + 1}/5 at {self._current_percentage:.1f}% - "
                          "Restarting...")
                    self._stop_download()
                    time.sleep(2)  # Brief pause before restart
                    break
                
                time.sleep(1)
        
        if self._download_completed:
            return True
        else:
            return self._fall_back_intervention()
    
    
    def _pull_with_speed_monitoring(self):
        """Pull with speed monitoring - placeholder"""
        print("Speed monitoring mode - placeholder implementation")
        # TODO: speed monitoring logic
        return self._pull_basic()
    
    
    def _pull_with_auto(self):
        """Pull with auto detection - placeholder"""
        print("Auto mode - placeholder implementation")
        # TODO: auto detection logic
        return self._pull_basic()
    
    
    def _pull_basic(self):
        """Basic pull without interventions"""
        self._start_download()
        
        while self._is_pulling:
            time.sleep(1)
        
        return self._download_completed
    
    
    def _start_download(self):
        """
        Start download in a separate thread
        """
        self._should_stop = False
        self._download_completed = False
        self._download_thread = threading.Thread(target=self._download_worker, daemon=True)
        self._download_thread.start()
    
    
    def _stop_download(self):
        """Stop the download thread"""
        self._should_stop = True
        if self._download_thread and self._download_thread.is_alive():
            self._download_thread.join(timeout=5)
    
    
    def _download_worker(self):
        """
        Worker function that runs in the thread
        """
        try:
            self._is_pulling = True
            stream = self.ollama_client.pull(self.model_name, stream=True)
            
            for chunk in stream:
                if self._should_stop:
                    break
                    
                if chunk.get("status"):
                    status = chunk["status"]
                    
                    if "pulling" in status.lower():
                        if 'completed' in chunk and 'total' in chunk:
                            completed = chunk['completed']
                            total = chunk['total']
                            self._current_percentage = (completed / total) * 100
                            self._last_completed = completed
                            
                            # Convert to MB for display
                            completed_mb = completed / (1024 * 1024)
                            total_mb = total / (1024 * 1024)
                            
                            print(f"\rProgress: {self._current_percentage:.1f}% "
                                  f"({completed_mb:.1f}MB/{total_mb:.1f}MB)", 
                                  end='', flush=True)
                    
                    elif 'success' in status.lower():
                        self._download_completed = True
                        print(f"\nModel {self.model_name} pulled successfully!")
                        break
                    
                    else:
                        print(f"\r{status}", end="", flush=True)
        
        except Exception as e:
            print(f"\nDownload error: {e}")
        finally:
            self._is_pulling = False
       
       
    def _fall_back_intervention(self):
        """
        Fallback method to pull the model with set interventions
        """
        print(f"Starting final download attempt with {self.fall_back_interval}s interventions...")
        max_attempts = 10

        for attempt in range(max_attempts):
            print(f"{self.fall_back_interval}s interval attempt {attempt + 1}/{max_attempts}")
            self._start_download()
            
            for _ in range(5):
                time.sleep(self.fall_back_interval/5)
                # Check if completed
                if self._download_completed:
                    print("Download completed!")
                    return True

            print(f"{self.fall_back_interval}s interval reached, restarting...")
            self._stop_download()
            time.sleep(2)

        return False
    
    
    def get_intervention_log(self):
        """Get log of completed interventions"""
        return self._completed_interventions