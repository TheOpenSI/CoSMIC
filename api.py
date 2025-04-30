from datetime import datetime
from fastapi import FastAPI
from src.opensi_cosmic import OpenSICoSMIC
from pydantic import BaseModel
import yaml, os, shutil
import pandas as pd
from zoneinfo import ZoneInfo

app = FastAPI()

class CosmicAPI(BaseModel):
    body: dict
    user_message: str

config_path = "scripts/configs/config_updated.yaml"
statistic_dir = "data/cosmic/statistic"
statistic_dict = {
            "user_id": "unknown",
            "email": "unknown",
            "start_date": -1,
            "last_date": -1,
            "average_token_length": 0,
            "query_count": 0
        }
openai_api_key = os.environ.get("OPENAI_API_KEY", "0p3n-w3bu!")

def update_statistic_table(statistic_dict):
    global statistic_dir

    os.makedirs(statistic_dir, exist_ok=True)
    current_time = statistic_dict["last_date"]
    time_split = current_time.split(",")[0].split("-")
    current_month_year = f"{time_split[1]}-{time_split[2]}"
    statistic_path = os.path.join(
        statistic_dir,
        f"{current_month_year}.csv"
    )

    if os.path.exists(statistic_path):
        data = pd.read_csv(statistic_path)
        user_emails = data["email"].tolist()
    else:
        user_emails = []

    if statistic_dict["email"] in user_emails:
        idx = [idx for idx, user_email in enumerate(user_emails) if user_email == statistic_dict["email"]][0]
        data.loc[idx, "last_date"] = statistic_dict["last_date"]
        history_total_token_length = data["average_token_length"][idx] * data["query_count"][idx]
        current_total_token_length = statistic_dict["average_token_length"] * statistic_dict["query_count"]
        total_query_count = data["query_count"][idx] + statistic_dict["query_count"]
        data.loc[idx, "average_token_length"] = (history_total_token_length + current_total_token_length) / total_query_count
        data.loc[idx, "query_count"] = total_query_count
    else:
        df = pd.DataFrame([{
            "user_id": statistic_dict["user_id"],
            "email": statistic_dict["email"],
            "start_date": statistic_dict["start_date"],
            "last_date": statistic_dict["last_date"],
            "average_token_length": statistic_dict["average_token_length"],
            "query_count": statistic_dict["query_count"]
        }])
        if len(user_emails) > 0: df = pd.concat([data, df], axis=0)
        data = df

    data.to_csv(
        statistic_path,
        header=[
            "user_id",
            "email",
            "start_date",
            "last_date",
            "average_token_length",
            "query_count"
        ],
        index=False
    )

def update_statistic_per_query(
        query,
        user_id,
        user_email,
        current_time
    ):
        global statistic_dict
        if False:
            # Save for the previous user (when the user_id changed).
            pre_user_id = statistic_dict["user_id"]
            pre_query_count = statistic_dict["query_count"]
            token_length = len(query)

            # Accumulate for the same user.
            if pre_user_id == user_id:
                pre_average_token_length = statistic_dict["average_token_length"]
                statistic_dict["last_date"] = current_time
                statistic_dict["average_token_length"] = \
                    (pre_average_token_length * pre_query_count + token_length) \
                    / (pre_query_count + 1)
                statistic_dict["query_count"] = pre_query_count + 1

            if pre_user_id != user_id:
                # Save previous user statistic.
                if pre_user_id != "unknown":
                    update_statistic_table(statistic_dict)

                # Initialize for a different user.
                statistic_dict["user_id"] = user_id
                statistic_dict["email"] = user_email
                statistic_dict["start_date"] = current_time
                statistic_dict["last_date"] = current_time
                statistic_dict["average_token_length"] = token_length
                statistic_dict["query_count"] = 1
        else:
            # Save every query for the current user.
            token_length = len(query)
            statistic_dict["user_id"] = user_id
            statistic_dict["email"] = user_email
            statistic_dict["start_date"] = current_time
            statistic_dict["last_date"] = current_time
            statistic_dict["average_token_length"] = token_length
            statistic_dict["query_count"] = 1
            update_statistic_table(statistic_dict)

def update_openai_key():
    global openai_api_key
    # Set up OPENAI_API_KEY globally through root's .env.
    openai_api_key = os.environ.get("OPENAI_API_KEY", "0p3n-w3bu!")

if not os.path.exists(config_path):
    shutil.copyfile("scripts/configs/config.yaml", config_path)

config_modify_timestamp = str(os.path.getmtime(config_path))

# Check if vector database is valid.
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

if not os.path.exists(config["rag"]["vector_db_path"]):
    config["rag"]["vector_db_path"] = "data/cosmic/vector_db_cosmic"

with open(config_path, "w") as file:
    yaml.safe_dump(config, file)

# Initialize the OpenSICoSMIC class
opensi_cosmic = OpenSICoSMIC(config_path=config_path)

openai_api_status = opensi_cosmic.check_openai_key()

@app.get("/")
async def read_root():
    return {"message": "Welcome to the OpenSICoSMIC API"}
    
@app.get("/quit")
async def quit():
    try:
        opensi_cosmic.quit()
        return {"status": "success", "message": "Application is shutting down"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    
@app.post("/cosmic")
async def process_cosmic(data: CosmicAPI):
    global config_modify_timestamp
    global openai_api_key
    global opensi_cosmic
    global openai_api_status
    try:
        current_config_modify_timestamp = str(os.path.getmtime(config_path))
        current_openai_api_key = os.environ.get("OPENAI_API_KEY", "0p3n-w3bu!")

        if (current_config_modify_timestamp != config_modify_timestamp) \
            or (current_openai_api_key != openai_api_key):
            opensi_cosmic.quit()
            openai_api_key = current_openai_api_key
            config_modify_timestamp = current_config_modify_timestamp
            os.environ["OPENAI_API_KEY"] = openai_api_key
            opensi_cosmic = OpenSICoSMIC(config_path=config_path)
            print('Reconstruct OpenSICoSMIC due to changed configs.')
            update_openai_key()
            openai_api_status = opensi_cosmic.check_openai_key()

        # Extract user_id from body. Adjust if user_id is available elsewhere.
        user_id = data.body["user"]["id"]
        user_role = data.body["user"]["role"]
        user_email = data.body["user"]["email"]

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
                file_dir = f"data/cosmic/backend/uploads/{user_id}"

                # Extract the files.
                files = splits[0].split("<files>")[-1]
                files = [os.path.join(file_dir, v) for v in files.split(',') if v != ""]

                for file in files:
                    # Form a prompt to update vector database.
                    user_message_vector_db_update = \
                        f"Add the following file to the vector database: {file}"

                    # Update vector database.
                    # answer = opensi_cosmic(user_message_vector_db_update)[0]
                    answer = opensi_cosmic(user_message_vector_db_update)[0]

            answer = opensi_cosmic(data.user_message)[0]
        return {"status": "success", "result": answer}
    except Exception as e:
        return {"status": "error", "message": str(e)}
