# -------------------------------------------------------------------------------------------------------------
# File: code_parser.py
# Project: OpenSI AI System
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

def parse_input(response: str):
    """
    Parse the input response to extract requirements, code, and example.
    Args:
        response (str): raw response from the model.
    """
    requirements = []
    code = ""
    example = ""

    if "### Answer" in response or "### Corrected Code" in response:
        return ["none"], response, ""
    
    lines = response.strip().splitlines()

    # A flag to keep track of the current section - requirements, code, or example.
    current_section = None

    # Depending on the output format, the response may contain multiple sections.
    # The flag current_section is used to keep track of the parsed output destination.
    for line in lines:
        if "### Requirements" in line:
            current_section = "requirements"
        elif "### Code" in line:
            current_section = "code"
        elif "### Example" in line:
            current_section = "example"
        else:
            if current_section == "requirements":
                if line.strip() not in ["bash", "```", ""]:
                    requirement = line.strip("`").strip().lower()
                    requirements.append(requirement)
            elif current_section == "code" or current_section == "example":
                # skip lines with triple backticks
                if line.strip() not in ["```", "```python"]:
                    if current_section == "code":
                        code += line + "\n"
                    elif current_section == "example":
                        # Sometimes LLMs generate the example to be used in a different file.
                        # In that case, the example will contain an import statement.
                        if "import" not in line.strip():
                            example += line + "\n"
    
    code = code.strip()
    example = example.strip()
    
    return requirements, code, example