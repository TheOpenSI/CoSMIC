# -------------------------------------------------------------------------------------------------------------
# File: log_tool.py
# Project: Open Source Institute-Cognitive System of Machine Intelligent Computing (OpenSI-CoSMIC)
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

# =============================================================================================================

class BCOLORS:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

# =============================================================================================================

INFOR_DICT = {
    "success": {"color": BCOLORS.OKGREEN, "comment": "Success"},
    "fail": {"color": BCOLORS.FAIL, "comment": "Fail"},
    "warning": {"color": BCOLORS.WARNING, "comment": "Warning"},
    "info": {"color": BCOLORS.HEADER, "comment": "Info"},
    "error": {"color": BCOLORS.FAIL, "comment": "Error"},
}

# =============================================================================================================

def set_color(
    status: str,
    information: str
):
    """Set color to display information on terminal.

    Args:
        status (str): information type, see INFOR_DICT.keys.
        information (str): information to be printed.

    Returns:
        information (str): colorized information.
    """
    status = status.lower()

    return f"{INFOR_DICT[status]['color']}[{INFOR_DICT[status]['comment']}]{BCOLORS.ENDC} {information}"