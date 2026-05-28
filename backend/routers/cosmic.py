### Core modules ###
from os import environ
from pathlib import Path
from shutil import copyfile, copyfileobj
from yaml import safe_load, safe_dump
from datetime import datetime, timezone
from dotenv import dotenv_values, set_key
from fastapi import Request, File, UploadFile, HTTPException, APIRouter, status
from pydantic import BaseModel
from zoneinfo import ZoneInfo
import httpx
from codecarbon import EmissionsTracker

### Type hints ###
from typing import Any, List, Optional

### Internal modules ###
from ...src.opensi_cosmic import OpenSICoSMIC
from ...utils.chat_history import build_context_from_messages
from ...utils.general import validate_openai_api_key
from ...utils.log_tool import set_color
from ...utils.statistics import update_statistic_per_query

router = APIRouter()


UPLOAD_BASE_DIR: Path = (
    Path(__file__).resolve(strict=True).parent.parent / "third_party"
)
UPLOAD_BASE_DIR.mkdir(mode=0o777, parents=False, exist_ok=True)

EMISSIONS_DIR: Path = (
    Path(__file__).resolve(strict=True).parent.parent.parent / "data" / "emissions"
)
EMISSIONS_DIR.mkdir(parents=True, exist_ok=True)


config_path: Path = (
    Path(__file__).resolve(strict=True).parent.parent.parent
    / "scripts"
    / "configs"
    / "config_updated.yaml"
)


class Message(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class User(BaseModel):
    id: str
    role: str
    email: str


class Body(BaseModel):
    user: User
    messages: List[Message] = []


class CosmicAPI(BaseModel):
    chat_id: Optional[str] = None
    name: Optional[str] = "New chat"
    user_message: str
    body: Body


def update_openai_key():
    global openai_api_key

    openai_api_key = environ.get(
        key="OPENAI_API_KEY",
        default=dotenv_values(
            dotenv_path=".env",
            stream=None,
            verbose=False,
            interpolate=True,
            encoding="utf=8",
        ).get("OPENAI_API_KEY", ""),
    )

    if not openai_api_key:
        print(
            set_color(
                status="warning",
                information="OPENAI_API_KEY is required in .env or environment variables.",
            )
        )


# Initialize the OPENAI_API_KEY on startup.
update_openai_key()

if not config_path.exists(follow_symlinks=True):
    copyfile(
        src=(
            Path(__file__).resolve(strict=True).parent.parent.parent
            / "scripts"
            / "configs"
            / "config.yaml"
        ),
        dst=config_path,
        follow_symlinks=True,
    )

config_modify_timestamp = config_path.stat().st_mtime

# Check if vector database is valid.
with config_path.open(
    mode="r", buffering=-1, encoding="utf-8", errors=None, newline=None
) as file:
    config = safe_load(stream=file)

    if (
        not Path(config["rag"]["vector_db_path"])
        .resolve(strict=True)
        .exists(follow_symlinks=True)
    ):
        config["rag"]["vector_db_path"] = "data/vector_db_cosmic"

with config_path.open(
    mode="w", buffering=-1, encoding="utf-8", errors=None, newline=None
) as file:
    safe_dump(
        data=config,
        stream=file,
        default_style=None,
        default_flow_style=None,
        canonical=None,
        indent=None,
        width=None,
        allow_unicode=None,
        line_break=None,
        encoding="utf-8",
        explicit_start=None,
        explicit_end=None,
        version=None,
        tags=None,
        sort_keys=True,
    )

# Initialize the OpenSICoSMIC class
opensi_cosmic = OpenSICoSMIC(config_path=str(object=config_path))

# All these configs must be given to
# Open-WebUi/src/lib/components/admin/Settings/Configs.svelte;
# otherwise set them as Optional[the config name]=default value;


openai_api_status = opensi_cosmic.check_openai_key()


def rebuild_cosmic():
    """
    Rebuilds the OpenSICoSMIC instance if the configuration file or OpenAI API key changes.
    """
    global config_modify_timestamp
    global openai_api_key
    global opensi_cosmic
    global openai_api_status

    current_config_modify_timestamp = config_path.stat().st_mtime
    current_openai_api_key = environ.get(
        key="OPENAI_API_KEY",
        default=dotenv_values(
            dotenv_path=".env",
            stream=None,
            verbose=False,
            interpolate=True,
            encoding="utf=8",
        ).get("OPENAI_API_KEY", ""),
    )

    if (current_config_modify_timestamp != config_modify_timestamp) or (
        current_openai_api_key != openai_api_key
    ):
        opensi_cosmic.quit()
        update_openai_key()
        config_modify_timestamp = current_config_modify_timestamp
        opensi_cosmic = OpenSICoSMIC(config_path=str(object=config_path))
        print("Reconstruct OpenSICoSMIC due to changed configs.")
        openai_api_status = opensi_cosmic.check_openai_key()


@router.post("")
async def process_cosmic(data: CosmicAPI):
    global config_modify_timestamp
    global openai_api_key
    global opensi_cosmic
    global openai_api_status

    try:
        rebuild_cosmic()

        # Extract user_id from body. Adjust if user_id is available elsewhere.
        user_id = data.body.user.id
        user_role = data.body.user.role
        user_email = data.body.user.email

        # Chat history context.
        messages = [{"role": m.role, "content": m.content} for m in data.body.messages]
        chat_history_context = build_context_from_messages(messages, num_pairs=5)

        # Check if Chat History is empty.
        chat_history_template: str = "{0:s}\n\n{1:s}".format(
            "Conversation History:",
            "=============== End of Chat History ===============",
        )

        chat_history_context: str = (
            ""
            if chat_history_context.strip() == chat_history_template
            else chat_history_context
        )

        # Compute statistic information.
        current_time = datetime.strftime(
            datetime.now(tz=ZoneInfo("Australia/Sydney")), "%d-%m-%Y,%H:%M:%S"
        )

        update_statistic_per_query(
            query=[data.user_message],
            user_id=user_id,
            user_email=user_email,
            current_time=current_time,
        )

        # Proceed as normal
        if openai_api_status != "":
            answer: str = str(object=openai_api_status)
            return {
                "status": "success",
                # "condition": "no API key found",
                "result": answer,
            }

        else:
            # Find the key word for adding file to vector database.
            if data.user_message.find("</files>") > -1:
                splits: list[Any] = data.user_message.split("</files>")

                # Extract the original question.
                data.user_message = splits[1]

                # The directory storing uploaded files.
                file_dir: Path = (
                    Path(__file__).resolve(strict=True).parent.parent.parent
                    / "data"
                    / "upload"
                    / f"{user_id}"
                )

                # Extract the files.
                extracted_files: str = splits[0].split("<files>")[-1]
                new_files: list[str] = []

                for extracted_file in extracted_files.split(sep=",", maxsplit=-1):
                    if extracted_file != "":
                        new_files.append(str(object=file_dir.joinpath(extracted_file)))

                for new_file in new_files:
                    # Form a prompt to update vector database.
                    user_message_vector_db_update: str = (
                        f"Add the following file to the vector database: {new_file}"
                    )

                    tracker = EmissionsTracker(
                        project_name="cosmic-chat",
                        save_to_file=True,
                        output_dir=str(EMISSIONS_DIR),
                    )
                    tracker.start()

                    # Update vector database.
                    answer: str = str(
                        object=opensi_cosmic(question=user_message_vector_db_update)[0]
                    )
                    emissions = tracker.stop()
                    print(f"[CodeCarbon] VectorDB update emissions: {emissions:.8f} kg CO₂")

            else:

                tracker = EmissionsTracker(
                    project_name="cosmic-chat",
                    save_to_file=True,
                    output_dir=str(EMISSIONS_DIR),
                )
                tracker.start()


                answer: str = str(
                    object=opensi_cosmic(
                        question=data.user_message, context=chat_history_context
                    )[0]
                )

                emissions = tracker.stop()
                print(f"[CodeCarbon] Chat query emissions: {emissions:.8f} kg CO₂")


                # smanile - connect to database repo + return result
                CHAT_API_URL = "http://cosmic-backend-fastapi:8000/api/v1/chatboxes/"  # TODO: later when have time, move to .env file

                now = datetime.now(timezone.utc).isoformat()
                new_detail = {
                    "user_role": "user",
                    "user_query": data.user_message,
                    "query_create_on": now,
                    "llm_role": "assistant",
                    "llm_response": answer,
                    "response_create_on": now,
                }
                payload = {
                    "user_id": str(user_id),
                    "name": data.name,
                    "details": [new_detail],
                }

                async with httpx.AsyncClient() as client:
                    if data.chat_id:
                        save_response = await client.patch(
                            f"{CHAT_API_URL}{data.chat_id}",
                            json=payload,
                        )
                        save_response.raise_for_status()

                        chat_id = data.chat_id
                    else:
                        save_response = await client.post(
                            CHAT_API_URL,
                            json=payload,
                        )
                        save_response.raise_for_status()
                        chat_id = save_response.json()["created"]["id"]

                return {
                    "status": "success",
                    # "condition": "API key found",
                    "result": answer,
                    "chat_id": chat_id,
                }

    except HTTPException as http_exc:
        raise http_exc

    except Exception as fastapi_err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(object=fastapi_err),
        )
