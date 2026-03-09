# -------------------------------------------------------------------------------------------------------------
# File: root.py
# Project: Open Source Institute-Cognitive System of Machine Intelligent Computing (OpenSI-CoSMIC)
# Contributors:
#     Bing Tran <binhsan1307@gmail.com>
#
# Copyright (c) 2026 Open Source Institute
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


### Core modules ###
from fastapi import APIRouter, status


### Type hints ###


### Internal modules ###


root_router: APIRouter = APIRouter(
    tags=["Root Endpoints"],
    responses={
        200: {
            "description": "OK (Root Endpoints)"
        },
        405: {
            "description": "Method Not Allowed (Root Endpoints)"
        }
    }
)


@root_router.get(
    path="/",
    status_code=status.HTTP_200_OK,
    response_model=None
)
async def read_root() -> dict[str, str]:
    return {
        "message": "Welcome to CoSMIC project!",
        "version": "0.1"
    }


@root_router.post(
    path="/",
    status_code=status.HTTP_405_METHOD_NOT_ALLOWED,
    response_model=None
)
async def create_root() -> dict[str, int | str]:
    return {
        "status": status.HTTP_405_METHOD_NOT_ALLOWED,
        "message": "POST request method is not allowed for Root Endpoints"
    }


@root_router.patch(
    path="/",
    status_code=status.HTTP_405_METHOD_NOT_ALLOWED,
    response_model=None
)
async def update_root() -> dict[str, int | str]:
    return {
        "status": status.HTTP_405_METHOD_NOT_ALLOWED,
        "message": "PATCH request method is not allowed for Root Endpoints"
    }


@root_router.delete(
    path="/",
    status_code=status.HTTP_405_METHOD_NOT_ALLOWED,
    response_model=None
)
async def delete_root() -> dict[str, int | str]:
    return {
        "status": status.HTTP_405_METHOD_NOT_ALLOWED,
        "message": "DELETE request method is not allowed for Root Endpoints"
    }
