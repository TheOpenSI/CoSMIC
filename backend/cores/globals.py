### Core modules ###
from pathlib import Path
from shutil import copyfile
from yaml import (
    safe_load,
    safe_dump
)
from dotenv import dotenv_values


### Type hints ###
from typing import Any


### Internal modules ###


# TODO:these 3 will be deleted soon
OG_CONFIG_PATH: Path = Path(__file__).resolve(strict=True).parent.parent.parent.joinpath(
    "scripts",
    "configs",
    "config.yaml"
)
NEW_CONFIG_PATH: Path = OG_CONFIG_PATH.parent.joinpath("config_updated.yaml")
ENV_PATH: Path = Path(__file__).resolve(strict=True).parent.parent.parent.joinpath(".env")
CORS_ALLOW_ORIGIN: list[str] = [
    "http://localhost:5173",    # FE binded Docker port
    "http://localhost:8000",    # BE binded Docker port
    "http://localhost:3000",    # CoSMIC binded Docker port
    "http://localhost:11434"    # Ollama binded Docker port
]
CORS_ALLOW_METHODS: list[str] = [
    "POST",
    "PATCH",
    "GET",
    "DELETE",
    "OPTIONS"
]
CORS_ALLOW_HEADERS: list[str] = []
CORS_REQUEST_TIMEOUT: int = 600


# TODO:these 2 will be deleted soon
def config_healthcheck() -> None:
    if not NEW_CONFIG_PATH.exists(follow_symlinks=True):
        copyfile(
            src=OG_CONFIG_PATH,
            dst=NEW_CONFIG_PATH,
            follow_symlinks=True
        )

    else:
        with NEW_CONFIG_PATH.open(
            mode="r",
            encoding="utf-8"
        ) as config_file:
            config: dict[str, Any] = safe_load(config_file)

            if not Path(config["rag"]["vector_db_path"]).resolve(strict=True).exists(follow_symlinks=True):
                config["rag"]["vector_db_path"] = "data/vector_db_cosmic"

            else:
                # Do nothing here since path is valid and exists
                pass

        with NEW_CONFIG_PATH.open(
            mode="w",
            encoding="utf-8"
        ) as config_file:
            safe_dump(
                data=config,
                stream=config_file,
                encoding="utf-8",
                sort_keys=False # Respect key-value order in original config YAML file
            )

    return None


def load_openai_key() -> str | None:
    api_key: dict[str, Any] = dotenv_values(
        dotenv_path=ENV_PATH,
        encoding= "utf-8"
    )

    return api_key.get(
        "OPENAI_API_KEY",
        None
    )
