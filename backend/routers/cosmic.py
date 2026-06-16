### Core modules ###
from os import environ
from pathlib import Path
from datetime import (
    datetime,
    timezone
)
from dotenv import dotenv_values
from fastapi import (
    Depends,
    Request,
    HTTPException,
    APIRouter,
    status
)
from pydantic import BaseModel
from httpx import (
    AsyncClient,
    Response
)
import csv
from codecarbon import EmissionsTracker
from ...utils.log_tool import set_color


### Type hints ###
from typing import Any


### Internal modules ###
from ..cores.dependencies import (
    get_config_path,
    get_openai_status,
    get_opensi_cosmic
)
from ...src.opensi_cosmic import OpenSICoSMIC
from ...src.services.document_index import DocumentIndex
from ...utils.chat_history import build_context_from_messages
# from ...utils.statistics import update_statistic_per_query
from . import memory


router: APIRouter = APIRouter()


EMISSIONS_PATH: Path = Path(__file__).resolve(strict=True).parent.parent.parent.joinpath(
    "data",
    "emissions"
)
EMISSIONS_PATH.mkdir(
    mode=0o777,
    parents=True,
    exist_ok=True
)


# TODO:
# there'll be a new way to format this new chat session payload data from FE that
# we don't need to rely on the legacy data format anymore as soon as we get to
# work on the migration from old to new `/config` endpoint. For more information,
# refer to `api.py` (for the new endpoint) & `default_apis.py` (for the mentioned
# of new changes).
class Message(BaseModel):
    role:       str  # "user" | "assistant"
    content:    str


class User(BaseModel):
    id:     str
    role:   str
    email:  str


class Body(BaseModel):
    user:       User
    messages:   list[Message | None] = []


class CosmicAPI(BaseModel):
    chat_id:        str | None = None
    name:           str | None = "New chat"
    user_message:   str
    body:           Body

# Initialize document indexes for metadata tracking
global_document_index = DocumentIndex(persist_path="/app/data/memories/global/global_document_index.json")
user_document_index = DocumentIndex(persist_path="/app/data/memories/users/user_document_index.json")
memory.set_document_indexes(global_document_index, user_document_index)

def add_request_context_to_latest_emission(
    user_id: str,
    chat_id: str | None
) -> None:
    """Find the latest individual emission file, add user/chat id, merge into master CSV."""
    master_file: Path = EMISSIONS_PATH.joinpath("emissions.csv")

    # Find all individual emission files (not the master)
    individual_files: list[Path] = sorted(
        [f for f in EMISSIONS_PATH.glob("emission_*.csv")],
        key=lambda f: f.stat().st_mtime
    )

    if not individual_files:
        return None

    # Get the most recent one — that's the current query
    latest_file: Path = individual_files[-1]

    with latest_file.open(
        mode="r",
        encoding="utf-8",
        newline=""
    ) as csv_file:
        reader:     csv.DictReader[str]     = csv.DictReader(csv_file)
        rows:       list[dict[Any, Any]]    = list(reader)
        fieldnames: list[str | None]        = list(reader.fieldnames or [])

    # Remove blank rows
    rows: list[dict[Any, Any]] = [
        row
        for row in rows
        if any(
            v.strip()
            for v in row.values()
        )
    ]

    if not rows:
        latest_file.unlink()  # delete empty file
        return None

    # Add user_id and chat_id
    for extra_column in ("user_id", "chat_id"):
        if extra_column not in fieldnames:
            fieldnames.append(extra_column)

    for row in rows:
        row["user_id"] = str(user_id)
        row["chat_id"] = (
            ""
            if   (chat_id is None)
            else (chat_id)
        )

    # Append to master CSV
    master_exists: bool = master_file.exists(follow_symlinks=True)

    with master_file.open(
        mode="a",
        encoding="utf-8",
        newline=""
    ) as csv_file:
        writer: csv.DictWriter[str | None] = csv.DictWriter(
            f=csv_file,
            fieldnames=fieldnames
        )

        if not master_exists:
            writer.writeheader()

        writer.writerows(rows)

    # Delete the individual file after merging
    latest_file.unlink()


# TODO:
# Refer to the note on new changes in this same endpoint but in the legacy file
# (`default_apis.py`) for future updates.
@router.post("")
async def process_cosmic(
    data:               CosmicAPI,
    request:            Request,
    opensi_cosmic:      OpenSICoSMIC    = Depends(get_opensi_cosmic),
    openai_api_status:  str             = Depends(get_openai_status),
    config_path:        Path            = Depends(get_config_path)
):
    try:
        # Rebuild if config or API key changed
        current_ts = config_path.stat().st_mtime
        current_key = environ.get(
            "OPENAI_API_KEY",
            dotenv_values(".env").get(
                "OPENAI_API_KEY",
                ""
            )
        )

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

        user_id:    str     = data.body.model_dump(mode="json")["user"]["id"]
        user_role:  str     = data.body.model_dump(mode="json")["user"]["role"]
        # user_email: str     = data.body.model_dump(mode="json")["user"]["email"]

        chat_history_context: str = build_context_from_messages(
            messages=data.body.model_dump(mode="json").get(
                "messages",
                []
            ),
            num_pairs=5
        )

        chat_history_template: str = (
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
                "memories",
                "users",
                f"{user_id}"
            )
            if data.chat_id:
                file_dir = file_dir.joinpath(
                  "sessions",
                  f"{data.chat_id}"
                )

            # Extract the files.
            extracted_files: str = splits[0].split("<files>")[-1]
            new_files: list[str] = [
                str(file_dir.joinpath(extracted_file))
                for extracted_file in extracted_files.split(",")
                if  extracted_file
            ]

            for new_file in new_files:
                # Form a prompt to update vector database
                user_message_vector_db_update: str = f"Add the following file to the vector database: {new_file}"

                # Start CodeCarbon emission tracking process (for RAG-triggered user queries)
                rag_query_time: str = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S%f")
                rag_tracker: EmissionsTracker = EmissionsTracker(
                    project_name="cosmic-chat",
                    save_to_file=True,
                    output_dir=str(EMISSIONS_PATH),
                    output_file=f"emission_{rag_query_time}.csv",  # unique file per query
                    allow_multiple_runs=True,
                    log_level="error"
                )

                rag_tracker.start()

                # Update vector database
                answer: str = str(opensi_cosmic(question=user_message_vector_db_update)[0])

                # Stop CodeCarbon emission tracking process (for RAG-triggered
                # user queries) and start saving those tracked data
                rag_emissions: float | None = rag_tracker.stop()
                add_request_context_to_latest_emission(
                    user_id=user_id,
                    chat_id=data.chat_id
                )
                print(
                    set_color(
                        status="info",
                        information=f"[CodeCarbon] VectorDB update emissions: {rag_emissions:.8f} kg CO₂"
                    )
                )

        else:
            # Start CodeCarbon emission tracking process (for General user queries)
            general_query_time: str = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S%f")
            general_tracker: EmissionsTracker = EmissionsTracker(
                project_name="cosmic-chat",
                save_to_file=True,
                output_dir=str(EMISSIONS_PATH),
                output_file=f"emission_{general_query_time}.csv",  # unique file per query
                allow_multiple_runs=True,
                log_level="error"
            )

            general_tracker.start()

            answer: str = str(
                opensi_cosmic(
                    question=data.user_message,
                    context=chat_history_context,
                )[0]
            )

            # TODO: later when have time, move to '.env' file
            CHAT_API_URL: str = "http://backend:8000/api/v1/chatboxes/"

            now: str = datetime.now(tz=timezone.utc).isoformat()
            new_detail: dict[str, Any] = {
                "user_role":            user_role,
                "user_query":           data.user_message,
                "query_create_on":      now,
                "llm_role":             "assistant",
                "llm_response":         answer,
                "response_create_on":   now,
            }
            payload: dict[str, Any] = {
                "user_id":  str(user_id),
                "name":     data.name,
                "details":  [new_detail],
            }

            async with AsyncClient() as client:
                if data.chat_id:
                    save_response: Response = await client.patch(
                        f"{CHAT_API_URL}{data.chat_id}",
                        json=payload
                    )
                    save_response.raise_for_status()
                    chat_id: str = data.chat_id

                else:
                    save_response: Response = await client.post(
                        CHAT_API_URL,
                        json=payload
                    )
                    save_response.raise_for_status()
                    chat_id = save_response.json()["created"]["id"]

            # Stop CodeCarbon emission tracking process (for General user queries)
            # and start saving those tracked data
            general_emissions: float | None = general_tracker.stop()
            add_request_context_to_latest_emission(
                user_id=user_id,
                chat_id=data.chat_id
            )
            print(
                set_color(
                    status="info",
                    information=f"[CodeCarbon] General chat query emissions: {general_emissions:.8f} kg CO₂"
                )
            )

            return {
                "status": "success",
                "result": answer,
                "chat_id": chat_id
            }


    except HTTPException as http_exc:
        raise http_exc


    except Exception as fastapi_err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"{fastapi_err}"
        )
