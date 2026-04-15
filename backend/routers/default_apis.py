### Core modules ###
from os import environ
from pathlib import Path
from shutil import (
    copyfile,
    copyfileobj
)
from yaml import (
    safe_load,
    safe_dump
)
from datetime import datetime
from dotenv import (
    dotenv_values,
    set_key
)
from fastapi import (
    Request,
    File,
    UploadFile,
    HTTPException,
    APIRouter,
    status
)
from pydantic import BaseModel
from zoneinfo import ZoneInfo


### Type hints ###
from typing import Any


### Internal modules ###
from ...src.opensi_cosmic import OpenSICoSMIC
from ...utils.chat_history import build_context_from_messages
from ...utils.general import validate_openai_api_key
from ...utils.log_tool import set_color
from ...utils.statistics import update_statistic_per_query


router = APIRouter()


UPLOAD_BASE_DIR: Path = (Path(__file__).resolve(strict=True).parent.parent / "third_party")
UPLOAD_BASE_DIR.mkdir(
    mode=0o777,
    parents=False,
    exist_ok=True
)


config_path: Path = (Path(__file__).resolve(strict=True).parent.parent.parent / "scripts" / "configs" / "config_updated.yaml")


class CosmicAPI(BaseModel):
    body: dict
    user_message: str


def update_openai_key():
    global openai_api_key

    openai_api_key = environ.get(
        key="OPENAI_API_KEY",
        default=dotenv_values(
            dotenv_path=".env",
            stream=None,
            verbose=False,
            interpolate=True,
            encoding="utf=8"
        ).get(
            "OPENAI_API_KEY",
            ""
        )
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
        src=(Path(__file__).resolve(strict=True).parent.parent.parent / "scripts" / "configs" / "config.yaml"),
        dst=config_path,
        follow_symlinks=True
    )

config_modify_timestamp = config_path.stat().st_mtime

# Check if vector database is valid.
with config_path.open(
    mode="r",
    buffering=-1,
    encoding="utf-8",
    errors=None,
    newline=None
) as file:
    config = safe_load(stream=file)

    if not Path(config["rag"]["vector_db_path"]).resolve(strict=True).exists(follow_symlinks=True):
        config["rag"]["vector_db_path"] = "data/vector_db_cosmic"

with config_path.open(
    mode="w",
    buffering=-1,
    encoding="utf-8",
    errors=None,
    newline=None
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
        sort_keys=True
    )

# Initialize the OpenSICoSMIC class
opensi_cosmic = OpenSICoSMIC(config_path=str(object=config_path))

# All these configs must be given to
# Open-WebUi/src/lib/components/admin/Settings/Configs.svelte;
# otherwise set them as Optional[the config name]=default value;


class QueryQnalyserConfig(BaseModel):
    llm_name: str = opensi_cosmic.config_data["query_analyser"]["llm_name"]
    is_quantized: bool = opensi_cosmic.config_data["query_analyser"]["is_quantized"]


class RAGConfig(BaseModel):
    top_k: int = opensi_cosmic.config_data["rag"]["topk"]
    retrieve_score_threshold: float = opensi_cosmic.config_data["rag"]["retrieve_score_threshold"]
    vector_db_path: str = opensi_cosmic.config_data["rag"]["vector_db_path"]


class ChessConfig(BaseModel):
    stockfish_path: str = opensi_cosmic.config_data["chess"]["stockfish_path"]


class OpenAIConfig(BaseModel):
    api_key: str | None = openai_api_key


class ConfigUpdateForm(BaseModel):
    llm_name: str
    is_quantized: bool
    seed: int
    doc_directory: str
    document_path: str
    # service: list[int] # TODO: Change to a list of integers.
    service: int
    sameasabove: bool = False
    query_analyser: QueryQnalyserConfig
    rag: RAGConfig
    chess: ChessConfig
    openai: OpenAIConfig


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
            encoding="utf=8"
        ).get(
            "OPENAI_API_KEY",
            ""
        )
    )

    if (current_config_modify_timestamp != config_modify_timestamp) \
    or (current_openai_api_key != openai_api_key):
        opensi_cosmic.quit()
        update_openai_key()
        config_modify_timestamp = current_config_modify_timestamp
        opensi_cosmic = OpenSICoSMIC(config_path=str(object=config_path))
        print("Reconstruct OpenSICoSMIC due to changed configs.")
        openai_api_status = opensi_cosmic.check_openai_key()


@router.get("/config")
async def get_config():
    try:
        with open(config_path, "r") as file:  # was config_default_path
            config_data = safe_load(stream=file)
        config_data["OPENAI_API_KEY"] = openai_api_key

        return config_data

    except HTTPException as http_exc:
        raise http_exc

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/config/update")
async def update_config(
    request: Request,
    form_data: ConfigUpdateForm
):
    try:
        # Update all the variables to /app/backend/configs/config_default.yaml.
        # This will be used by CoSMIC pipeline.
        # Step 1: read config.yaml
        with config_path.open(
            mode="r",
            buffering=-1,
            encoding="utf-8",
            errors=None,
            newline=None
        ) as file: # was config_default_path
            config_data = safe_load(stream=file)

        # Step 2: update the values, and save it as config_default.yaml.
        config_data["llm_name"]                         = form_data.llm_name
        config_data["is_quantized"]                     = form_data.is_quantized
        config_data["seed"]                             = form_data.seed
        config_data["service"]                          = form_data.service
        config_data["query_analyser"]["llm_name"]       = form_data.query_analyser.llm_name
        config_data["query_analyser"]["is_quantized"]   = form_data.query_analyser.is_quantized
        config_data["rag"]["topk"]                      = form_data.rag.top_k
        config_data["rag"]["retrieve_score_threshold"]  = form_data.rag.retrieve_score_threshold
        config_data["sameasabove"]                      = form_data.sameasabove

        # Update only when these paths exist, instead of overwriting by invalid paths.
        if Path(form_data.doc_directory).resolve(strict=True).exists(follow_symlinks=True):
            config_data["doc_directory"] = form_data.doc_directory

        if Path(form_data.document_path).resolve(strict=True).exists(follow_symlinks=True):
            config_data["document_path"] = form_data.document_path

        if Path(form_data.rag.vector_db_path).resolve(strict=True).exists(follow_symlinks=True):
            config_data["rag"]["vector_db_path"] = form_data.rag.vector_db_path

        if Path(form_data.chess.stockfish_path).resolve(strict=True).exists(follow_symlinks=True):
            config_data["chess"]["stockfish_path"] = form_data.chess.stockfish_path

        # Save OpenAI API key to .env instead of displaying in config_updated.yaml.
        env_path = ".env"

        is_llm_name_gpt = form_data.llm_name.find("gpt") > -1
        is_query_analyser_llm_name_gpt = form_data.query_analyser.llm_name.find("gpt") > -1

        # This might not be useful as it is in docker container.

        if is_llm_name_gpt or is_query_analyser_llm_name_gpt:
            if not form_data.openai.api_key == "" \
            and validate_openai_api_key(form_data.openai.api_key):
                environ["OPENAI_API_KEY"] = form_data.openai.api_key

                # Change the root's .env which is shared with .env in this backend container.
                set_key(
                    dotenv_path=env_path,
                    key_to_set="OPENAI_API_KEY",
                    value_to_set=form_data.openai.api_key,
                    quote_mode="always",
                    export=False,
                    encoding="utf-8",
                    follow_symlinks=False
                )

                update_openai_key()

            else:
                raise HTTPException(
                    status_code=400, detail="Invalid OpenAI API key provided."
                )

        # Save updated configs to config_updated.yaml, instead of overwriting config.yaml.
        with config_path.open(
            mode="w",
            buffering=-1,
            encoding="utf-8",
            errors=None,
            newline=None
        ) as file:
            safe_dump(
                data=config_data,
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
                sort_keys=True
            )

        # return request.app.config

        # rebuild_cosmic()

        return {
            "status": "success",
            "message": "Configuration updated successfully"
        }

    except HTTPException as http_exc:
        raise http_exc

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chess/upload")
async def upload_file(
        file: UploadFile = File(...)
):
    try:
        if file.content_type != "application/octet-stream":
            raise HTTPException(
                status_code=400, detail="Only binary files are allowed."
            )

        folder_name = file.filename
        save_dir = UPLOAD_BASE_DIR.joinpath(str(object=folder_name))
        save_dir.mkdir(
            mode=0o777,
            parents=False,
            exist_ok=True
        )

        save_path = save_dir.joinpath(str(object=file.filename))

        with save_path.open(
            mode="wb",
            buffering=-1,
            encoding="utf-8",
            errors=None,
            newline=None
        ) as buffer:
            copyfileobj(
                fsrc=file.file,
                fdst=buffer
            )

        with config_path.open(
            mode="r",
            buffering=-1,
            encoding="utf-8",
            errors=None,
            newline=None
        ) as file: # was config_default_path
            config_data = safe_load(stream=file)

        config_data["chess"]["stockfish_path"] = save_path

        with config_path.open(
            mode="w",
            buffering=-1,
            encoding="utf-8",
            errors=None,
            newline=None
        ) as file:
            safe_dump(
                data=config_data,
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
                sort_keys=True
            )

        return {
            "status": "success",
            "message": f"File saved to {save_path}"
        }

    except HTTPException as http_exc:
        raise http_exc

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/quit")
async def quit():
    try:
        opensi_cosmic.quit()
        return {
            "status": "success",
            "message": "Application is shutting down"
        }

    except HTTPException as http_exc:
        raise http_exc

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cosmic")
async def process_cosmic(data: CosmicAPI):
    global config_modify_timestamp
    global openai_api_key
    global opensi_cosmic
    global openai_api_status

    try:
        rebuild_cosmic()

        # Extract user_id from body. Adjust if user_id is available elsewhere.
        user_id = data.body["user"]["id"]
        user_role = data.body["user"]["role"]
        user_email = data.body["user"]["email"]

        # Chat history context.
        chat_history_context = build_context_from_messages(
            messages=data.body.get("messages", []),
            num_pairs=5
        )
        # Check if Chat History is empty.
        chat_history_template: str = "{0:s}\n\n{1:s}".format(
            "Conversation History:",
            "=============== End of Chat History ==============="
        )

        chat_history_context: str = ""                                  \
            if chat_history_context.strip() == chat_history_template    \
            else chat_history_context

        # Compute statistic information.
        current_time = datetime.strftime(
            datetime.now(tz=ZoneInfo("Australia/Sydney")), "%d-%m-%Y,%H:%M:%S"
        )

        update_statistic_per_query(
            query=[data.user_message],
            user_id=user_id,
            user_email=user_email,
            current_time=current_time
        )

        # Proceed as normal
        if openai_api_status != "":
            answer: str = str(object=openai_api_status)
            return {
                "status": "success",
                "result": ""
            }

        else:
            # Proceed as normal
            if openai_api_status != "":
                answer: str = openai_api_status
                return {
                    "status": "success",
                    "result": answer
                }

            else:
                # Find the key word for adding file to vector database.
                if data.user_message.find("</files>") > -1:
                    splits: list[Any] = data.user_message.split("</files>")

                    # Extract the original question.
                    data.user_message = splits[1]

                    # The directory storing uploaded files.
                    file_dir: Path = (Path(__file__).resolve(strict=True).parent.parent.parent / "data" / "upload" / f"{user_id}")

                    # Extract the files.
                    extracted_files: str = splits[0].split("<files>")[-1]
                    new_files: list[str] = []

                    for extracted_file in extracted_files.split(sep=",", maxsplit=-1):
                        if extracted_file != "":
                            new_files.append(str(object=file_dir.joinpath(extracted_file)))

                    for new_file in new_files:
                        # Form a prompt to update vector database.
                        user_message_vector_db_update: str = f"Add the following file to the vector database: {new_file}"

                        # Update vector database.
                        answer: str = str(
                            object=opensi_cosmic(
                                question=user_message_vector_db_update
                            )[0]
                        )

                else:
                    answer: str = str(
                        object=opensi_cosmic(
                            question=data.user_message,
                            context=chat_history_context
                        )[0]
                    )

                    return {
                        "status": "success",
                        "result": answer
                    }

    except HTTPException as http_exc:
            raise http_exc

    except Exception as fastapi_err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(object=fastapi_err)
        )
