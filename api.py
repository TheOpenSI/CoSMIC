# =====================Debugging========================
# Uncomment the following lines to enable debugging
# import debugpy
# print("Waiting for debugger attach...")
# debugpy.listen(("0.0.0.0", 5678))
# debugpy.wait_for_client()
# print("Debugger attached!")
# =======================================================

from datetime import datetime
import dotenv
from fastapi import FastAPI, Request, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from numpy import int64
from src.opensi_cosmic import OpenSICoSMIC
from pydantic import BaseModel
import yaml, os, shutil
import pandas as pd
from zoneinfo import ZoneInfo
from typing import Optional

from utils.chat_history import build_context_from_messages
from utils.general import validate_openai_api_key
from utils.log_tool import set_color
from utils.statistics import update_statistic_per_query

app = FastAPI()

# To test CORS_ALLOW_ORIGIN locally, you can set something like
# CORS_ALLOW_ORIGIN=http://localhost:5173;http://localhost:8080
# in your .env file depending on your frontend port, 8080 or 5173 in this case.

CORS_ALLOW_ORIGIN = [
    "http://localhost:8080",  # Open-WebUI production server
    "http://localhost:5173",  # Open-WebUI development server
]

# Allow CORS for the specified origins
if os.environ.get("CORS_ALLOW_ORIGIN"):
    CORS_ALLOW_ORIGIN.extend(os.environ.get("CORS_ALLOW_ORIGIN").split(";"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGIN,  # or ["*"] to allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_BASE_DIR = "third_party"
os.makedirs(UPLOAD_BASE_DIR, exist_ok=True)

class CosmicAPI(BaseModel):
    body: dict
    user_message: str

config_path = "scripts/configs/config_updated.yaml"

def update_openai_key():
    global openai_api_key
    openai_api_key = os.environ.get("OPENAI_API_KEY", dotenv.dotenv_values(".env").get("OPENAI_API_KEY", ""))
    if not openai_api_key:
        print(set_color("warning", "OPENAI_API_KEY is required in .env or environment variables."))


# Initialize the OPENAI_API_KEY on startup.
update_openai_key()

if not os.path.exists(config_path):
    shutil.copyfile("scripts/configs/config.yaml", config_path)

config_modify_timestamp = str(os.path.getmtime(config_path))

# Check if vector database is valid.
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

if not os.path.exists(config["rag"]["vector_db_path"]):
    config["rag"]["vector_db_path"] = "backend/data/vector_db_cosmic"

with open(config_path, "w") as file:
    yaml.safe_dump(config, file)

# Initialize the OpenSICoSMIC class
opensi_cosmic = OpenSICoSMIC(config_path=config_path)

# All these configs must be given to
# Open-WebUi/src/lib/components/admin/Settings/Configs.svelte;
# otherwise set them as Optional[the config name]=default value;

class QueryQnalyserConfig(BaseModel):
    llm_name: Optional[str] = opensi_cosmic.config['query_analyser']['llm_name']
    is_quantized: Optional[bool] = opensi_cosmic.config['query_analyser']['is_quantized']


class RAGConfig(BaseModel):
    top_k: Optional[int] = opensi_cosmic.config['rag']['topk']
    retrieve_score_threshold: Optional[float] = opensi_cosmic.config['rag']['retrieve_score_threshold']
    vector_db_path: Optional[str] = opensi_cosmic.config['rag']['vector_db_path']


class ChessConfig(BaseModel):
    stockfish_path: Optional[str] = opensi_cosmic.config['chess']['stockfish_path']


class OpenAIConfig(BaseModel):
    api_key: Optional[str] = openai_api_key

class ConfigUpdateForm(BaseModel):
    llm_name: str
    is_quantized: bool
    seed: int
    doc_directory: str
    document_path: str
    # service: list[int] # TODO: Change to a list of integers.
    service: int
    sameasabove: Optional[bool] = False
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

    current_config_modify_timestamp = str(os.path.getmtime(config_path))
    current_openai_api_key = os.environ.get("OPENAI_API_KEY", dotenv.dotenv_values(".env").get("OPENAI_API_KEY", ""))

    if (current_config_modify_timestamp != config_modify_timestamp) \
            or (current_openai_api_key != openai_api_key):
        opensi_cosmic.quit()
        update_openai_key()
        config_modify_timestamp = current_config_modify_timestamp
        opensi_cosmic = OpenSICoSMIC(config_path=config_path)
        print('Reconstruct OpenSICoSMIC due to changed configs.')
        openai_api_status = opensi_cosmic.check_openai_key()

@app.get("/")
async def read_root():
    return {"message": "Welcome to the OpenSICoSMIC API"}

@app.get("/config")
async def get_config():
    try:
        with open(config_path, "r") as file:  # was config_default_path
            config_data = yaml.safe_load(file)
        config_data["OPENAI_API_KEY"] = openai_api_key

        return config_data
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/config/update")
async def update_config(request: Request, form_data: ConfigUpdateForm):
    try:
        # Update all the variables to /app/backend/configs/config_default.yaml.
        # This will be used by CoSMIC pipeline.
        # Step 1: read config.yaml
        with open(config_path, "r") as file:  # was config_default_path
            config_data = yaml.safe_load(file)

        # Step 2: update the values, and save it as config_default.yaml.
        config_data["llm_name"] = form_data.llm_name
        config_data["is_quantized"] = form_data.is_quantized
        config_data["seed"] = form_data.seed
        config_data["service"] = form_data.service
        config_data["query_analyser"]["llm_name"] = form_data.query_analyser.llm_name
        config_data["query_analyser"]["is_quantized"] = form_data.query_analyser.is_quantized
        config_data["rag"]["topk"] = form_data.rag.top_k
        config_data["rag"]["retrieve_score_threshold"] = form_data.rag.retrieve_score_threshold
        config_data["sameasabove"] = form_data.sameasabove

        # Update only when these paths exist, instead of overwriting by invalid paths.
        if os.path.exists(form_data.doc_directory):
            config_data["doc_directory"] = form_data.doc_directory

        if os.path.exists(form_data.document_path):
            config_data["document_path"] = form_data.document_path

        if os.path.exists(form_data.rag.vector_db_path):
            config_data["rag"]["vector_db_path"] = form_data.rag.vector_db_path

        if os.path.exists(form_data.chess.stockfish_path):
            config_data["chess"]["stockfish_path"] = form_data.chess.stockfish_path

        # # Save OpenAI API key to .env instead of displaying in config_updated.yaml.
        env_path = ".env"

        is_llm_name_gpt = form_data.llm_name.find("gpt") > -1
        is_query_analyser_llm_name_gpt = form_data.query_analyser.llm_name.find("gpt") > -1

        # This might not be useful as it is in docker container.

        if is_llm_name_gpt or is_query_analyser_llm_name_gpt:
            if not form_data.openai.api_key == "" and validate_openai_api_key(form_data.openai.api_key):
                os.environ["OPENAI_API_KEY"] = form_data.openai.api_key
                # Change the root's .env which is shared with .env in this backend container.
                dotenv.set_key(env_path, "OPENAI_API_KEY", form_data.openai.api_key)
                update_openai_key()
            else:
                raise HTTPException(status_code=400, detail="Invalid OpenAI API key provided.")

        # Save updated configs to config_updated.yaml, instead of overwriting config.yaml.
        with open(config_path, "w") as file:
            yaml.safe_dump(config_data, file)

        # return request.app.config
        # rebuild_cosmic()
        return {"status": "success", "message": "Configuration updated successfully"}
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/chess/upload")
async def upload_file(file: UploadFile = File(...)):
    try:

        if file.content_type != "application/octet-stream":
            raise HTTPException(status_code=400, detail="Only binary files are allowed.")

        folder_name = file.filename
        save_dir = os.path.join(UPLOAD_BASE_DIR, folder_name)
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, file.filename)

        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        with open(config_path, "r") as file:  # was config_default_path
            config_data = yaml.safe_load(file)

        config_data["chess"]["stockfish_path"] = save_path

        with open(config_path, "w") as file:
            yaml.safe_dump(config_data, file)

        return {"status": "success", "message": f"File saved to {save_path}"}
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/quit")
async def quit():
    try:
        opensi_cosmic.quit()
        return {"status": "success", "message": "Application is shutting down"}
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/cosmic")
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
            data.body.get("messages", []),
            num_pairs=5
        )
        # Check if Chat History is empty.
        chat_history_context = "" \
            if chat_history_context.strip() == 'Conversation History: \n\n=============== End of Chat History ===============' \
            else chat_history_context
                               
        # Set user ID to use a specific vector database.
        # For the same user, the QA instance will not change.
        opensi_cosmic.set_up_qa(str(user_id))

        # Compute statistic information.
        current_time = datetime.strftime(
            datetime.now(tz=ZoneInfo("Australia/Sydney")),
            '%d-%m-%Y,%H:%M:%S'
        )

        update_statistic_per_query(
            data.user_message,
            user_id,
            user_email,
            current_time
        )

        # Proceed as normal
        if openai_api_status != "":
            answer = openai_api_status
        else:
            # Find the key word for adding file to vector database.
            if data.user_message.find("</files>") > -1:
                splits = data.user_message.split("</files>")

                # Extract the original question.
                data.user_message = splits[1]
                

                # The directory storing uploaded files.
                file_dir = f"backend/data/uploads/{user_id}"

                # Extract the files.
                files = splits[0].split("<files>")[-1]
                files = [os.path.join(file_dir, v) for v in files.split(',') if v != ""]

                for file in files:
                    # Form a prompt to update vector database.
                    user_message_vector_db_update = \
                        f"Add the following file to the vector database: {file}"

                    # Update vector database.
                    answer = opensi_cosmic(user_message_vector_db_update)[0]

            answer = opensi_cosmic(question=data.user_message,
                                   context=chat_history_context)[0]
        return {"status": "success", "result": answer}
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
