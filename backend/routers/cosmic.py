### Core modules ###
from pathlib import Path
from datetime import (
    datetime,
    timezone
)

from uuid import UUID
from dotenv import dotenv_values
from fastapi import (
    Depends,
    HTTPException,
    APIRouter,
    Request,
    status
)
from pydantic import BaseModel
from httpx import (
    AsyncClient,
    Response
)
from codecarbon import EmissionsTracker
from ...utils.log_tool import set_color


### Type hints ###
from typing import Any


### Internal modules ###
from ..cores.dependencies import get_opensi_cosmic
from ...src.opensi_cosmic import OpenSICoSMIC
from ...src.services.document_index import DocumentIndex
from ...utils.chat_history import build_context_from_messages
from . import memory
# from ...utils.statistics import update_statistic_per_query


router: APIRouter = APIRouter()


# TODO:
# there'll be a new way to format this new chat session payload data from FE that
# we don't need to rely on the legacy data format anymore as soon as we get to
# work on the migration from old to new `/config` endpoint.
#
# UPDATE:
# This should be done in a sepearate PR instead.

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


async def send_emissions_to_db(
    user_id: str,
    tracker: EmissionsTracker,
) -> None:
    """Send emissions data directly via POST request to database."""
    ed = tracker.final_emissions_data

    if ed is None:
        print(set_color(status="error", information="[Emissions DB] No emissions data available"))
        return

    emission_payload: dict[str, Any] = {
        "run_id":                   ed.run_id,
        "duration":                 ed.duration,
        "emissions":                ed.emissions,
        "emissions_rate":           ed.emissions_rate,
        "cpu_power":                ed.cpu_power,
        "gpu_power":                ed.gpu_power,
        "ram_power":                ed.ram_power,
        "cpu_energy":               ed.cpu_energy,
        "gpu_energy":               ed.gpu_energy,
        "ram_energy":               ed.ram_energy,
        "energy_consumed":          ed.energy_consumed,
        "water_consumed":           ed.water_consumed,
        "region":                   ed.region,
        "cloud_provider":           ed.cloud_provider,
        "cloud_region":             ed.cloud_region,
        "os":                       ed.os,
        "cpu_count":                ed.cpu_count,
        "cpu_model":                ed.cpu_model,
        "gpu_count":                ed.gpu_count,
        "gpu_model":                ed.gpu_model,
        "longitude":                ed.longitude,
        "latitude":                 ed.latitude,
        "ram_total_size":           ed.ram_total_size,
        "tracking_mode":            ed.tracking_mode,
        "cpu_utilization_percent":  ed.cpu_utilization_percent,
        "gpu_utilization_percent":  ed.gpu_utilization_percent,
        "ram_utilization_percent":  ed.ram_utilization_percent,
        "ram_used_gb":              ed.ram_used_gb,
        "on_cloud":                 ed.on_cloud,
        "pue":                      ed.pue,
        "wue":                      ed.wue,
        "user_id":                  str(user_id),
    }

    EMISSIONS_API_URL: str = "http://backend:8000/api/v1/emissions/"

    try:
        async with AsyncClient() as client:
            response: Response = await client.post(
                EMISSIONS_API_URL,
                json=emission_payload
            )
            response.raise_for_status()
            print(set_color(
                status="info",
                information=f"[Emissions DB] Successfully stored emissions for user {user_id}"
            ))
    except Exception as e:
        print(set_color(
            status="error",
            information=f"[Emissions DB] Failed to store emissions: {str(e)}"
        ))


@router.post("")
async def process_cosmic(
    data:               CosmicAPI,
    opensi_cosmic:      OpenSICoSMIC = Depends(get_opensi_cosmic)
):
    try:
        user_id:    str     = data.body.model_dump(mode="json")["user"]["id"]
        # user_role:  str     = data.body.model_dump(mode="json")["user"]["role"]
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

        openai_api_status: str = opensi_cosmic.check_openai_key()

        if openai_api_status != "":
            return {
                "status": "success",
                "result": openai_api_status
            }

        # Proceed as normal
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
                    save_to_file=False, 
                    allow_multiple_runs=True,
                    tracking_mode="process",  # track only the current process (not the whole machine)
                    log_level="error"
                )

                rag_tracker.start()

                # Update vector database
                answer: str = str(
                    opensi_cosmic(
                        question=user_message_vector_db_update,
                        session_id=data.chat_id,
                        has_files=True,
                        user_id=user_id,
                    )[0]
                )

                # Stop CodeCarbon emission tracking process (for RAG-triggered
                # user queries) and start saving those tracked data
                rag_emissions: float | None = rag_tracker.stop()
                await send_emissions_to_db(
                    user_id=user_id,
                    tracker=rag_tracker
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
                save_to_file=False,  
                allow_multiple_runs=True,
                tracking_mode="process",  # track only the current process (not the whole machine)
                log_level="error"
            )

            general_tracker.start()

            answer: str = str(
                opensi_cosmic(
                    question=data.user_message,
                    context=chat_history_context,
                    session_id=data.chat_id,
                    has_files=False,
                    user_id=user_id,
                )[0]
            )

            # TODO: later when have time, move to '.env' file
            CHAT_API_URL: str = "http://backend:8000/api/v1/chatboxes/"

            now: str = datetime.now(tz=timezone.utc).isoformat()
            new_detail: dict[str, Any] = {
                "user_role":            "user",         # per agreed solution within our team
                "user_query":           data.user_message,
                "query_create_on":      now,
                "llm_role":             "assistant",    # per agreed solution within our team
                "llm_response":         answer,
                "response_create_on":   now,
                # TODO: later we have to implement the function to calculate the number of
                # tokens used for each query and response, and then store it in the database. 
                # For now, we will just set it to 1.
                "input_token":          1,
                "output_token":         1,
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
            await send_emissions_to_db(
                user_id=user_id,
                tracker=general_tracker
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


    except Exception as fastapi_exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"{fastapi_exc}"
        )


@router.patch("")
async def rebuild_cosmic(
    request:        Request,
    opensi_cosmic:  OpenSICoSMIC = Depends(get_opensi_cosmic)
):
    try:
        latest_configs: list[dict[str, dict[str, Any]]] = opensi_cosmic.get_configs()

        if not latest_configs:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "status": "503 - API Services Unavailable",
                    "message:": "Could not reach Configurations API."
                }
            )

        else:
            # Similar trick to prevent having to perform expensive for-loop. Take a
            # look at `src/opensi_cosmic.py` [Line 85]
            latest_config_data:             dict[str, Any] = list(latest_configs[0].values())[0]

            latest_general_config_data:     dict[str, Any] = latest_config_data["general"]
            default_general_config_data:    dict[str, Any] = request.app.state.default_configs["general"]

            latest_qa_config_data:          dict[str, Any] = latest_config_data["query_analyser"]
            default_qa_config_data:         dict[str, Any] = request.app.state.default_configs["query_analyser"]

            if  (latest_general_config_data == default_general_config_data) \
            and (latest_qa_config_data == default_qa_config_data):
                # Accidentally click 'Save' button? No worries, nothing will
                # happens except a friendly "warning" meesage :)
                return {
                    "status": "success",
                    "message": "`OpenSICoSMIC()` stay the same due to exact config data found during update request."
                }

            else:
                # Clear cached memories from currently used SLMs first
                opensi_cosmic.quit()

                # Then re-state it again to allow `OpenSICoSMIC()` receiving new config data
                request.app.state.opensi_cosmic     = OpenSICoSMIC()
                request.app.state.default_configs   = latest_config_data

                return {
                    "status": "success",
                    "message": "`OpenSICoSMIC()` reconstructing due to new config data found during update request..."
                }


    except HTTPException as http_exc:
        raise http_exc


    except Exception as fastapi_err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"{fastapi_err}"
        )
