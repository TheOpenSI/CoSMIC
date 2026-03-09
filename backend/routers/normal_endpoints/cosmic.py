# -------------------------------------------------------------------------------------------------------------
# File: cosmic.py
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
# NOTE: These will be deleted once I fully refactored the current code
# from ...todo.api import rebuild_cosmic
# from ...todo.src.opensi_cosmic import OpenSICoSMIC
# from ...todo.utils.chat_history import build_context_from_messages


cosmic_router: APIRouter = APIRouter(
    prefix="/cosmic",
    tags=["Cosmic Endpoints"],
    responses={
        200: {
            "description": "OK (Cosmic Endpoints)"
        },
        201: {
            "description": "Created (Cosmic Endpoints)"
        },
        404: {
            "description": "Not Found (Cosmic Endpoints)"
        },
        405: {
            "description": "Method Not Allowed (Cosmic Endpoints)"
        }
    }
)


@cosmic_router.get(
    path="/",
    status_code=status.HTTP_200_OK,
    response_model=None
)
async def read_cosmic() -> dict[str, str]:
    return {
        "message": "This is the cosmic page of CoSMIC project!"
    }


@cosmic_router.post(
    path="/",
    status_code=status.HTTP_201_CREATED,
    response_model=None
)
async def create_cosmic() -> dict[str, str]:
    return {
        "status": "WIP",
        "progress": "5%"
    }
    # ###  TODO: I got no clue what this does but just get it works for now ###
    # ### before proper refactor.                                           ###
    # opensi_cosmic: OpenSICoSMIC = OpenSiCoSMIC()
    # ###  TODO: I got no clue what this does but just get it works for now ###
    # ### before proper refactor.                                           ###
    # openai_api_status: str = opensi_cosmic.check_openai_key()
    # # openai_api_key: str = ""
    # # config_modify_timestamp: str = ""
    #
    #
    # ### TODO: Delete this and moved into its own logic and file instead. ###
    # rebuild_cosmic()
    #
    #
    # ### TODO: Hard-coded for now as deadline rush, no time to look yet. ###
    # # Extract user_id from body. Adjust if user_id is available elsewhere.
    # user_id:    str = "123456789"
    # user_name:  str = "CoSMIC"
    # # user_role:  str = "User"
    # # user_email: str = "opensi@canberra.edu.au"
    #
    #
    # # Check if chat history is empty or not
    # chat_history_context: str = build_context_from_messages(
    #     messages=["message"],
    #     num_pairs=5
    # )
    # chat_history_check_format: str = "{0:s}{1:s}".format(
    #     "=============== Conversation History ===============\n",
    #     "=============== End of Chat History ===============\n"
    # )
    #
    # if (chat_history_context.strip() == chat_history_check_format):
    #     chat_history_context = ""
    # else:
    #     pass
    #
    #
    # # Set user ID to use a specific vector database.
    # # For the same user, the QA instance will not change.
    # opensi_cosmic.set_up_qa(
    #     user_id=user_id,
    #     user_name=user_name
    # )
    #
    #
    # ### TODO: Refactor this code last as it adds data to Statistics API ###
    # #
    # # # Compute statistic information.
    # # current_time = datetime.strftime(
    # #     datetime.now(tz=ZoneInfo("Australia/Sydney")),
    # #     '%d-%m-%Y,%H:%M:%S'
    # # )
    # #
    # # update_statistic_per_query(
    # #     data.user_message,
    # #     user_id,
    # #     user_email,
    # #     current_time
    # # )
    #
    #
    # # Proceed as normal
    # if (openai_api_status != ""):
    #     answer: str = openai_api_status
    #     return {
    #         "status": "success",
    #         "result": answer
    #     }
    # else:
    #     return {
    #         "status": "failed",
    #         "result": ""
    #     }
    #
    #
    #     ### TODO: Delete as well, maybe a better way to find, append, and ###
    #     ### grab data from file.                                          ###
    #     #
    #     # # Find the key word for adding file to vector database.
    #     # if data.user_message.find("</files>") > -1:
    #     #     splits = data.user_message.split("</files>")
    #     #
    #     #     # Extract the original question.
    #     #     data.user_message = splits[1]
    #     #
    #     #     # The directory storing uploaded files.
    #     #     file_dir = f"backend/data/uploads/{user_id}"
    #     #
    #     #     # Extract the files.
    #     #     files = splits[0].split("<files>")[-1]
    #     #     files = [os.path.join(file_dir, v) for v in files.split(',') if v != ""]
    #     #
    #     #     for file in files:
    #     #         # Form a prompt to update vector database.
    #     #         user_message_vector_db_update = \
    #     #             f"Add the following file to the vector database: {file}"
    #     #
    #     #         # Update vector database.
    #     #         answer = opensi_cosmic(user_message_vector_db_update)[0]
    #     #
    #     # answer = opensi_cosmic(
    #     #     question=data.user_message,
    #     #     context=chat_history_context
    #     # )[0]


@cosmic_router.patch(
    path="/",
    status_code=status.HTTP_405_METHOD_NOT_ALLOWED,
    response_model=None
)
async def update_cosmic() -> dict[str, int | str]:
    return {
        "status": status.HTTP_405_METHOD_NOT_ALLOWED,
        "message": "PATCH request method is not allowed for Cosmic Endpoints"
    }


@cosmic_router.delete(
    path="/",
    status_code=status.HTTP_405_METHOD_NOT_ALLOWED,
    response_model=None
)
async def delete_cosmic() -> dict[str, int | str]:
    return {
        "status": status.HTTP_405_METHOD_NOT_ALLOWED,
        "message": "DELETE request method is not allowed for Cosmic Endpoints"
    }
