### Core modules ###
from os import environ
from pathlib import Path
from shutil import copyfileobj
from yaml import (
    safe_load,
    safe_dump
)
from dotenv import (
    dotenv_values,
    set_key
)
from fastapi import (
    Depends,
    Request,
    File,
    UploadFile,
    HTTPException,
    APIRouter,
    status
)
from pydantic import BaseModel


### Type hints ###
from typing import Any


### Internal modules ###
from ..cores.dependencies import (
    get_config_path,
    get_openai_key,
    get_openai_status,
    get_opensi_cosmic
)
from ...src.opensi_cosmic import OpenSICoSMIC
from ...utils.chat_history import build_context_from_messages
from ...utils.general import validate_openai_api_key
# from ...utils.statistics import update_statistic_per_query


router: APIRouter = APIRouter()


# TODO:
# these 2 global vars will get deleted for the same reason mentioned at the very
# start of `api.py` file
UPLOAD_BASE_DIR: Path = Path(__file__).resolve(strict=True).parent.parent.parent.joinpath("third_party")
UPLOAD_BASE_DIR.mkdir(
    mode=0o777,
    parents=False,
    exist_ok=True
)


# TODO:
# This class will either be removed or modified to work with the new chatbox
# session payload data format received from FE
class CosmicAPI(BaseModel):
    body:           dict
    user_message:   str


# TODO:
# these 5 classes will get deleted for the same reason mentioned at the very
# start of `api.py` file
class QueryQnalyserConfig(BaseModel):
    llm_name:       str
    is_quantized:   bool


class RAGConfig(BaseModel):
    top_k:                      int
    retrieve_score_threshold:   float
    vector_db_path:             str


class ChessConfig(BaseModel):
    stockfish_path: str


class OpenAIConfig(BaseModel):
    api_key: str | None


class ConfigUpdateForm(BaseModel):
    llm_name:       str
    is_quantized:   bool
    seed:           int
    doc_directory:  str
    document_path:  str | dict[str, Any]
    # service:        list[int] # TODO: Change to a list of integers.
    service:        int
    sameasabove:    bool = False
    query_analyser: QueryQnalyserConfig
    rag:            RAGConfig
    chess:          ChessConfig
    openai:         OpenAIConfig



# TODO:
# these endpoints (except `/cosmic`) will get deleted soon for the same reason
# mentioned at the very start of `api.py` file
@router.get("/config")
async def get_config(
    openai_api_key:     str     = Depends(get_openai_key),
    config_path:        Path    = Depends(get_config_path)
):
    try:
        with config_path.open(
            mode="r",
            encoding="utf-8"
        ) as config_file:
            config_data: dict[str, Any] = safe_load(stream=config_file)

        config_data["OPENAI_API_KEY"] = openai_api_key

        return config_data


    except HTTPException as http_exc:
        raise http_exc


    except Exception as fastapi_err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"{fastapi_err}"
        )


@router.post("/config/update")
async def update_config(
    request:        Request,
    form_data:      ConfigUpdateForm,
    config_path:    Path    = Depends(get_config_path)
):
    try:
        with config_path.open(
            mode="r",
            encoding="utf-8"
        ) as config_file:
            config_data: dict[str, Any] = safe_load(stream=config_file)

        config_data["llm_name"]                         = form_data.llm_name
        config_data["is_quantized"]                     = form_data.is_quantized
        config_data["seed"]                             = form_data.seed
        config_data["service"]                          = form_data.service
        config_data["query_analyser"]["llm_name"]       = form_data.query_analyser.llm_name
        config_data["query_analyser"]["is_quantized"]   = form_data.query_analyser.is_quantized
        config_data["rag"]["topk"]                      = form_data.rag.top_k
        config_data["rag"]["retrieve_score_threshold"]  = form_data.rag.retrieve_score_threshold
        config_data["sameasabove"]                      = form_data.sameasabove

        if Path(form_data.doc_directory).resolve(strict=True).exists(follow_symlinks=True):
            config_data["doc_directory"] = form_data.doc_directory

        if Path(str(form_data.document_path)).resolve(strict=True).exists(follow_symlinks=True):
            config_data["document_path"] = form_data.document_path

        if Path(form_data.rag.vector_db_path).resolve(strict=True).exists(follow_symlinks=True):
            config_data["rag"]["vector_db_path"] = form_data.rag.vector_db_path

        if Path(form_data.chess.stockfish_path).resolve(strict=True).exists(follow_symlinks=True):
            config_data["chess"]["stockfish_path"] = form_data.chess.stockfish_path

        is_llm_name_gpt:                bool = form_data.llm_name.find("gpt") > -1
        is_query_analyser_llm_name_gpt: bool = form_data.query_analyser.llm_name.find("gpt") > -1

        if (is_llm_name_gpt) \
        or (is_query_analyser_llm_name_gpt):
            if  (form_data.openai.api_key) \
            and (validate_openai_api_key(api_key=form_data.openai.api_key)):
                environ["OPENAI_API_KEY"]           = form_data.openai.api_key
                request.app.state.openai_api_key    = form_data.openai.api_key  # keep state in sync

                set_key(
                    dotenv_path=Path(__file__).resolve(strict=True).parent.parent.parent.joinpath(".env"),
                    key_to_set="OPENAI_API_KEY",
                    value_to_set=form_data.openai.api_key,
                    encoding="utf-8",
                    follow_symlinks=False
                )

            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid OpenAI API key provided."
                )

        with config_path.open(
            mode="w",
            encoding="utf-8"
        ) as config_file:
            safe_dump(
                data=config_data,
                stream=config_file,
                encoding="utf-8",
                sort_keys=False # Respect key-value order in oroginal config YAML file
            )


        return {
            "status": "success",
            "message": "Configuration updated successfully"
        }


    except HTTPException as http_exc:
        raise http_exc


    except Exception as fastapi_err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"{fastapi_err}"
        )


@router.post("/chess/upload")
async def upload_file(
    file:           UploadFile  = File(...),
    config_path:    Path        = Depends(get_config_path),
):
    try:
        if file.content_type != "application/octet-stream":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only binary files are allowed."
            )

        else:
            save_dir: Path = UPLOAD_BASE_DIR.joinpath(str(file.filename))
            save_dir.mkdir(
                mode=0o777,
                parents=False,
                exist_ok=True
            )
            save_path: Path = save_dir.joinpath(str(file.filename))

            with save_path.open(
                mode="wb",
                encoding="utf-8"
            ) as buffer:
                copyfileobj(
                    fsrc=file.file,
                    fdst=buffer
                )

            with config_path.open(
                mode="r",
                encoding="utf-8"
            ) as config_file:
                config_data: dict[str, Any] = safe_load(stream=config_file)

            config_data["chess"]["stockfish_path"] = str(save_path)

            with config_path.open(
                mode="w",
                encoding="utf-8"
            ) as config_file:
                safe_dump(
                    data=config_data,
                    stream=config_file,
                    encoding="utf-8",
                    sort_keys=False # Respect key-value order in oroginal config YAML file
                )

            return {
                "status": "success",
                "message": f"File saved to {save_path}"
            }


    except HTTPException as http_exc:
        raise http_exc


    except Exception as fastapi_err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"{fastapi_err}"
        )


@router.get("/quit")
async def quit(
    opensi_cosmic: OpenSICoSMIC = Depends(get_opensi_cosmic)
):
    try:
        opensi_cosmic.quit()

        return {
            "status": "success",
            "message": "Application is shutting down"
        }


    except HTTPException as http_exc:
        raise http_exc


    except Exception as fastapi_err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"{fastapi_err}"
        )


# TODO:
# this endpoint will be moved into its own file (we already have it), and its
# logic will get modified to work with the new chatbox session payload data
# received from FE. Eventually, when the migration is finished, this legacy
# endpoint will be removed and we will announce clients to use the new endpoint
# instead.
@router.post("/cosmic")
async def process_cosmic(
    data:               CosmicAPI,
    request:            Request,
    opensi_cosmic:      OpenSICoSMIC    = Depends(get_opensi_cosmic),
    openai_api_status:  str             = Depends(get_openai_status),
    config_path:        Path            = Depends(get_config_path)
):
    try:
        # Rebuild if config or API key changed
        current_ts: float = config_path.stat().st_mtime
        current_key: str | None = environ.get(
            "OPENAI_API_KEY",
            dotenv_values(".env").get(
                "OPENAI_API_KEY",
                None
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

        user_id:    str     = data.body["user"]["id"]
        # user_role:  str     = data.body["user"]["role"]
        # user_email: str     = data.body["user"]["email"]

        chat_history_context: str = build_context_from_messages(
            messages=data.body.get("messages", []),
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
                if  extracted_file
            ]

            for new_file in new_files:
                answer: str = str(opensi_cosmic(question=f"Add the following file to the vector database: {new_file}")[0])
        else:
            answer: str = str(
                opensi_cosmic(
                    question=data.user_message,
                    context=chat_history_context,
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
            detail=f"{fastapi_err}"
        )
