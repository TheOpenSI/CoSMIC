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

import subprocess, os, shutil, sys
from utils.log_tool import set_color

class PyCapsule:
    def __init__(self, 
                 IMAGE_NAME: str = "ghost525/sandbox_python", 
                 container_name: str = "opensi_sandbox_service"):
        """Code generation entry.

        self.countainer_mount_path contains a shell script to run the main file.
        Users can change the container_mount_path if needed.

        Args:
            IMAGE_NAME (str, optional): Default.
            container_name (str, optional): Default.
        """
        self.IMAGE_NAME = IMAGE_NAME
        self.container_name = container_name

        # Creating the container mount directory.
        root = sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../..")
        self.container_mount_path = os.path.join(root, "results/code_generation/container_mount")
        os.makedirs(self.container_mount_path, exist_ok=True)

        # Copying the shell script to the container mount directory.
        source_bash_file = os.path.join(root, "scripts/code_generation/start.sh")
        destination_bash_file = os.path.join(self.container_mount_path, "start.sh")
        shutil.copyfile(source_bash_file, destination_bash_file)

        # Add -x to the shell script.
        subprocess.run(f"chmod +x {destination_bash_file}", shell=True)

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

        status = not containers.stdout.strip() == ""

        return status

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