# -------------------------------------------------------------------------------------------------------------
# File: pycapsule.py
# Project: OpenSI AI System
# Contributors:
#     Muntasir Adnan <adnan.adnan@canberra.edu.au>
#
# IMPORTANT:
# - This service will use docker containers
# - So user will have to be have permission to run docker commands
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

import subprocess, os, re, shutil
from utils.log_tool import set_color

class PyCapsule:
    def __init__(self, 
                 IMAGE_NAME: str = "ghost525/sandbox_python", 
                 container_name: str = "opensi_sandbox_service"):
        """
        Args:
            IMAGE_NAME (str, optional): Default.
            container_name (str, optional): Default.
            countainer_mount_path will be /results/container_mount/ which has a shell script 
            to run the main file.
            ** Users can change the container_mount_path if needed.
        """
        self.IMAGE_NAME = IMAGE_NAME
        self.container_name = container_name
        # Creating the container mount directory.
        os.makedirs(f"{os.path.abspath(__file__).replace('pycapsule.py', '')}../../results/container_mount", 
                    exist_ok=True)
        # Copying the shell script to the container mount directory.
        source_folder = os.path.abspath(__file__).replace('pycapsule.py', '')+"../../scripts/start.sh"
        destination_folder = os.path.abspath(__file__).replace('pycapsule.py', '')+"../../results/container_mount/start.sh"
        shutil.copyfile(source_folder, destination_folder)
        # Add -x to the shell script.
        subprocess.run(f"chmod +x {destination_folder}", shell=True)
        self.container_mount_path = re.sub(r"/src.*", "/results/container_mount", 
                                           os.path.abspath(__file__)) # chnage here to mount a different path.
        self.check_if_image_exists() # this will pull image if not found.

    def check_if_image_exists(self, 
                              image_url:str = "ghost525/sandbox_python"):
        """
        Args:
            image_url (str, optional): Defaults to "ghost525/sandbox_python".
        """
        images = subprocess.run(f"docker images | grep {self.IMAGE_NAME}", 
                                shell=True, 
                                capture_output=True, 
                                text=True)
        if images.stdout.strip() == "":
            print(set_color("info", "Image does not exist"))
            print(set_color("info", "Pulling image..."))
            subprocess.run(f"docker pull {image_url}", shell=True)
            print(set_color("success", "Image built"))
        else:
            print(set_color("success", "Image found"))
    
    def check_if_container_exists(self) -> bool:
        containers = subprocess.run(f"docker ps -a | grep {self.container_name}", 
                                    shell=True, 
                                    capture_output=True, 
                                    text=True)
        if containers.stdout.strip() == "":
             print(set_color("info", "Container does not exist"))
        else:
            print(set_color("info", "Container found"))

        return not containers.stdout.strip() == ""

    def create_container(self):
        """
        Create the container and run the shell script which will install requirements and run main.py.
        """
        print(set_color("info", "Creating container..."))
        command = (
            f"docker run --name {self.container_name} "
            f"-v {self.container_mount_path}:/usr/src/app {self.IMAGE_NAME}"
        )
        response = subprocess.run(command, shell=True)
        print(set_color("success", "Container created"))
        print(set_color("info", f"[PYCAPSULE-EXIT CODE] {response.returncode}"))
        return response.returncode, response.stdout, response.stderr
    
    def start_container(self):
        """
        Start the container and run the shell script which will install requirements and run main.py.
        """
        print(set_color("info", "Starting container..."))
        response = subprocess.run(f"docker start -i {self.container_name}", 
                                  shell=True, 
                                  capture_output=True, 
                                  text=True)
        print(set_color("info", f"[PYCAPSULE-EXIT CODE] {response.returncode}"))
        return response.returncode, response.stdout, response.stderr