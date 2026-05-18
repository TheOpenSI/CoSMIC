### Core modules ###
from datetime import datetime, timezone
from os import environ
from pathlib import Path
from dotenv import dotenv_values
from fastapi import (
    Request,
    HTTPException,
    APIRouter,
    status,
    Depends
)
from pydantic import BaseModel
from httpx import AsyncClient, Response


### Type hints ###
from typing import Any


### Internal modules ###
from ...src.opensi_cosmic import OpenSICoSMIC
from ..cores.dependencies import (
    get_cosmic,
    get_openai_status,
    get_config_path
)
from ...utils.chat_history import build_context_from_messages


router: APIRouter = APIRouter(
    prefix="/api/v1/cosmic",
    tags=["CoSMIC - V1"]
)


class Message(BaseModel):
    role:       str  # "user" | "assistant"
    content:    str


class User(BaseModel):
    id:     str
    role:   str
    email:  str


class Body(BaseModel):
    user:       User
    messages:   list[Message] = []


class CosmicAPI(BaseModel):
    chat_id:        str | None  = None
    name:           str | None  = "New chat"
    user_message:   str
    body:           Body


@router.post("")
async def process_cosmic(
    data:               CosmicAPI,
    request:            Request,
    opensi_cosmic:      OpenSICoSMIC    = Depends(get_cosmic),
    openai_api_status:  str             = Depends(get_openai_status),
    config_path:        Path            = Depends(get_config_path),
):
    try:
        # Rebuild if config or API key changed
        current_ts = config_path.stat().st_mtime
        current_key = environ.get("OPENAI_API_KEY", dotenv_values(".env").get("OPENAI_API_KEY", ""))

        if (current_ts != request.app.state.config_modify_timestamp) \
        or (current_key != request.app.state.openai_api_key):
            opensi_cosmic.quit()
            request.app.state.openai_api_key            = current_key
            request.app.state.config_modify_timestamp   = current_ts
            request.app.state.opensi_cosmic             = OpenSICoSMIC(config_path=str(config_path))
            request.app.state.openai_api_status         = request.app.state.opensi_cosmic.check_openai_key()
            opensi_cosmic                               = request.app.state.opensi_cosmic
            openai_api_status                           = request.app.state.openai_api_status
            print("Reconstructed OpenSICoSMIC due to changed configs.")

        user_id:    str     = data.body["user"]["id"]
        user_role:  str     = data.body["user"]["role"]
        # user_email: str     = data.body["user"]["email"]

        chat_history_context = build_context_from_messages(
            messages=data.body.get("messages", []),
            num_pairs=5,
        )

        chat_history_template = (
            "Conversation History:\n\n"
            "=============== End of Chat History ==============="
        )
        if chat_history_context.strip() == chat_history_template:
            chat_history_context = ""

        # current_time = datetime.strftime(
        #     datetime.now(tz=ZoneInfo("Australia/Sydney")), "%d-%m-%Y,%H:%M:%S"
        # )
        # update_statistic_per_query(
        #     query=[data.user_message],
        #     user_id=user_id,
        #     user_email=user_email,
        #     current_time=current_time,
        # )

        # Proceed as normal
        if openai_api_status != "":
            return {
                "status": "success",
                "result": f"{str(openai_api_status)}"
            }

        if data.user_message.find("</files>") > -1:
            splits: list[str] = data.user_message.split("</files>")
            data.user_message = splits[1]
            file_dir: Path = Path(__file__).resolve().parent.parent.parent.joinpath(
                "data",
                "upload",
                f"{user_id}"
            )
            extracted_files: str = splits[0].split("<files>")[-1]
            new_files: list[str] = [
                str(file_dir.joinpath(extracted_file))
                for extracted_file in extracted_files.split(",")
                if extracted_file
            ]
            for new_file in new_files:
                answer: str = str(
                    opensi_cosmic(
                        question=f"Add the following file to the vector database: {new_file}"
                    )[0]
                )
        else:
            answer: str = str(
                opensi_cosmic(
                    question=data.user_message,
                    context=chat_history_context,
                )[0]
            )

            # smanile - connect to database repo + return result
            CHAT_API_URL: str = "http://backend:8000/api/v1/chatboxes/" # TODO: later when have time, move to .env file

            now: str = datetime.now(tz=timezone.utc).isoformat()
            new_detail: dict[str, Any] = {
                "user_role": user_role,
                "user_query": data.user_message,
                "query_create_on": now,
                "llm_role": "assistant",
                "llm_response": answer,
                "response_create_on": now,
            }
            payload: dict[str, Any] = {
                "user_id": str(user_id),
                "name": data.name,
                "details": [new_detail],
            }

            async with AsyncClient() as client:
                if data.chat_id:
                    save_response: Response = await client.patch(
                        f"{CHAT_API_URL}{data.chat_id}",
                        json=payload,
                    )
                    save_response.raise_for_status()

                    chat_id: str = data.chat_id
                else:
                    save_response: Response = await client.post(
                        CHAT_API_URL,
                        json=payload,
                    )
                    save_response.raise_for_status()
                    chat_id = save_response.json()["created"]["id"]

            return {
                "status": "success",
                "result": answer,
                "chat_id": chat_id,
            }


    except HTTPException as http_exc:
        raise http_exc


    except Exception as fastapi_err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"{fastapi_err}"
        )
