### THIS IS A DUMPING FILE FOR GETTING `/cosmic` ENDPOINT WORKING WITH NEW ARCHITECTURE JUST HOW AS IT IS ###
import yaml
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Json
from dotenv import dotenv_values
from .src.opensi_cosmic import OpenSICoSMIC
from utils.log_tool import set_color


UPLOAD_BASE_DIR:        Path = (Path(__file__).resolve(strict=True).parent / "third_party")
CONFIG_BASE_DIR:        Path = (Path(__file__).resolve(strict=True).parent / "scripts" / "configs")
CONFIG_DEFAULT_PATH:    Path = (CONFIG_BASE_DIR / "config.yaml")
CONFIG_MODIFIED_PATH:   Path = (CONFIG_BASE_DIR / "config_updated.yaml")
OPENAI_API_DIR:         Path = (Path(__file__).resolve(strict=True).parent / "todo.env")


# Create the `third_party` directory
try:
    UPLOAD_BASE_DIR.mkdir(
        mode=511,
        parents=False,
        exist_ok=False
    )
    # print(f"Directory [{UPLOAD_BASE_DIR}] created successfully.")
except FileExistsError:
    print(f"Directory [{UPLOAD_BASE_DIR}] already exists.")

except PermissionError:
    print(f"Permission denied: Unable to create [{UPLOAD_BASE_DIR}].")

except Exception as err:
    print(f"An error occurred: [{err}].")


# Read OpenAI API key in `.env` file
api_default_key: str | None = dotenv_values(
    dotenv_path=OPENAI_API_DIR,
    encoding= "utf-8"
)["OPENAI_API_KEY"]


def update_openai_key() -> str | None:
    if api_default_key == "":
        return set_color(
            status="warning",
            information="[OPENAI_API_KEY] is required in .env or environment variables."
        )
    else:
        return api_default_key


def cosmic_todo_startup() -> None:
    # Initialize the OPENAI_API_KEY on startup.
    update_openai_key()

    if not Path(CONFIG_MODIFIED_PATH).exists():
        CONFIG_DEFAULT_PATH.copy(
            CONFIG_MODIFIED_PATH,
            follow_symlinks=True,
            preserve_metadata=True
        )

        return None
    else:
        # Check if vector database is valid.
        with Path(CONFIG_MODIFIED_PATH).open(
            mode="rt",
            buffering=-1,
            encoding="utf-8",
            errors=None,
            newline=None
        ) as read_config_file:
            config_data: dict[str, Any] = yaml.safe_load(stream=read_config_file)
            print(f"Config current data: {config_data}")

            if config_data["rag"]["vector_db_path"] == "":
                config_data["rag"]["vector_db_path"] = "data/vector_db_cosmic"

                with Path(CONFIG_MODIFIED_PATH).open(
                    mode="w",
                    buffering=-1,
                    encoding="utf-8",
                    errors=None,
                    newline=None
                ) as write_config_file:
                    yaml.safe_dump(
                        data=config_data, 
                        stream=write_config_file
                    )

                print(f"Config new data: {config_data}")
            else:
                print(f"Config old data: {config_data}")
                pass

        return None


# Get default config file timestamp. Not sure why needed but oh well, same code.
config_default_timestamp: float = CONFIG_MODIFIED_PATH.stat().st_mtime
# print(f"Default config file timestamp: [{config_modify_timestamp}]")


# Pass default config file over to our main CoSMIC class for internal works.
opensi_cosmic: OpenSICoSMIC = OpenSICoSMIC(
    config_path=str(
        object=CONFIG_MODIFIED_PATH
    )
)

# Make sure the API key is provided (not really sure why you have to explicitly
# call it here??)
openai_api_status: str = opensi_cosmic.check_openai_key()


# Payload received from user query
class CosmicAPI(BaseModel):
    body: Json
    msg: str


def rebuild_cosmic(opensi_cosmic: OpenSICoSMIC) -> str | None:
    """
    Rebuilds the OpenSICoSMIC instance if the configuration file or OpenAI API key changes.
    """
    config_modify_timestamp: float = CONFIG_MODIFIED_PATH.stat().st_mtime
    api_modify_key: str | None = dotenv_values(
        dotenv_path=OPENAI_API_DIR,
        encoding= "utf-8"
    )["OPENAI_API_KEY"]

    if (
        (config_modify_timestamp != config_default_timestamp)
        or
        (api_modify_key != api_default_key)
    ):
        opensi_cosmic.quit()
        update_openai_key()

        print('Reconstruct OpenSICoSMIC due to changed configs.')
        openai_api_status: str = opensi_cosmic.check_openai_key()

        return openai_api_status
    else:
        return None
