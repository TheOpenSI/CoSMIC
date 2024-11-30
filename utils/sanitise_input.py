# -------------------------------------------------------------------------------------------------------------
# File: sanitise_input.py
# Project: Open Source Institute-Cognitive System of Machine Intelligent Computing (OpenSI-CoSMIC)
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

# https://docs.python.org/3/library/shlex.html#shlex.quote

import re
import shlex
import keyword

def sanitise_input(input_string: str) -> str:
    """
    Sanitizes input string by escaping special characters and validating it.
    Input string can only contain alphanumeric characters, '.', '_', ':', and '-'.

    Args:
        input_string (str): Input string to be sanitized.
        pattern (str): Regex pattern to validate the input.

    Raises:
        ValueError: If input string contains invalid characters.

    Returns:
        str: Sanitized input string.
    """
    pattern = r"^[a-zA-Z0-9._:-]+$"
    
    # Validate the input against the provided pattern
    if not re.match(pattern, input_string):
        raise ValueError(f"Invalid input: {input_string}")
    
    # Escape the input for safe shell usage
    return shlex.quote(input_string)


def sanitise_variable_name(name: str) -> str:
    """
    Fixes a string to make it a valid Python variable name.

    Parameters:
    - name (str): The input string to sanitize.

    Returns:
    - str: A valid Python variable name.
    """
    # Replace invalid characters with underscores
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name)

    # Ensure the name doesn't start with a digit
    if name and name[0].isdigit():
        name = f"_{name}"

    # Ensure the name isn't a Python reserved keyword
    if keyword.iskeyword(name):
        name = f"{name}_var"

    return name