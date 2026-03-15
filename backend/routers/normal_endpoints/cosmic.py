### Core modules ###
from pathlib import Path
from fastapi import APIRouter, HTTPException, status
from datetime import datetime
from zoneinfo import ZoneInfo


### Type hints ###


### Internal modules ###
# NOTE: These will be deleted once I fully refactored the current code
from ...todo.todo import CosmicAPI, opensi_cosmic, openai_api_status
from ...todo.api import rebuild_cosmic
from ...todo.utils.chat_history import build_context_from_messages
from ...todo.utils.statistics import update_statistic_per_query


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
async def create_cosmic(
    payload: CosmicAPI
) -> dict[str, str]:
    try:
        rebuild_cosmic()

        # Extract user_id from body. Adjust if user_id is available elsewhere.
        user_id:    int = payload.body["user"]["id"]
        user_email: str = payload.body["user"]["email"]
        user_role:  str = payload.body["user"]["role"]

        # Chat history context.
        chat_history_context: str = build_context_from_messages(
            messages=payload.body["messages"],
            num_pairs=5
        )

        # Check if Chat History is empty.
        chat_history_empty: str = "{0:s}{1:s}{2:s}{3:s}".format(
            "Conversation History:\n",
            "Previous Conversation Pair\n",
            f"{"-"*30}\n",
            f"{"="*15} End of Chat History {"="*15}\n"
        )

        if chat_history_context.strip() == chat_history_empty:
            chat_history_context = ""

            return {
                "status": "error",
                "result": chat_history_context
            }
        else:
            file_paths: list[Path] = []
            final_answer: str = ""

            # Set user ID to use a specific vector database.
            # For the same user, the QA instance will not change.
            opensi_cosmic.set_up_qa(
                user_id=str(object=user_id),
                user_name=""
            )

            # Compute statistic information.
            current_time: str = datetime.now(
                tz=ZoneInfo(key="Australia/Sydney")
            ).strftime(
                format="%d-%m-%Y,%H:%M:%S"
            )

            update_statistic_per_query(
                query=payload.msg,
                user_id=user_id,
                user_email=user_email,
                current_time=current_time
            )

            # Proceed as normal
            if openai_api_status != "":
                final_answer = openai_api_status
            else:
                # Find the key word for adding file to vector database.
                if payload.msg.find("</files>") > -1:
                    message_splits: list[str] = payload.msg.split("</files>")

                    # Extract the original question.
                    payload.msg = message_splits[1]

                    # The directory storing uploaded files.
                    USER_UPLOAD_DIR: Path = (Path(__file__).resolve(strict=True).parent.parent.parent / "data" / "uploads" / f"{user_id}")

                    # Create user-specific's uploaded documents directory
                    try:
                        USER_UPLOAD_DIR.mkdir(
                            mode=0o777,
                            parents=False,
                            exist_ok=False
                        )
                        # print(f"Directory [{USER_UPLOAD_DIR}] created successfully.")
                    except FileExistsError as file_err:
                        raise file_err

                    except PermissionError as perm_err:
                        raise perm_err

                    except Exception as err:
                        raise err

                    # Extract the files.
                    files_part: str = message_splits[0].split("<files>")[-1]
                    # Split by comma or newline to get individual files
                    file_list: list[str] = [file_part.strip() for file_part in files_part.split(",") if file_part.strip()]

                    for file in file_list:
                        file_paths.append(USER_UPLOAD_DIR / "file")

                        # Form a prompt to update vector database.
                        result = opensi_cosmic(
                            question=f"Add the following file to the vector database: {file}"
                        )

                        final_answer += str(object=result[0]) if result else ""

                    # Get the answer for the actual question
                    result: tuple = opensi_cosmic(
                        question=payload.msg,
                        context=chat_history_context
                    )

                    final_answer += str(object=result[0]) if result else ""
                else:
                    # Get the answer for the actual question
                    result: tuple = opensi_cosmic(
                        question=payload.msg,
                        context=chat_history_context
                    )

                    print("QA Debugging: So we know that QA works, and I'm not going to test the other services.")
                    final_answer += str(object=result[0]) if result else ""

            return {
                "status": "success",
                "result": final_answer
            }

    except HTTPException as http_exc:
        raise http_exc

    except Exception as err:
        raise HTTPException(
            status_code=500,
            detail=str(object=err)
        )


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
