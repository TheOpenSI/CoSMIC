### Core modules ###
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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
from .routers import (
    models,
    cosmic,
    default_apis
)
from .src.opensi_cosmic import OpenSICoSMIC
from .utils.log_tool import set_color


CONFIG_PATH: Path = Path(__file__).resolve(strict=True).parent.joinpath(
    "scripts",
    "configs",
    "config_updated.yaml"
)
ENV_PATH: Path = Path(__file__).resolve(strict=True).parent.joinpath(
    ".env"
)
CORS_ALLOW_ORIGIN: list[str] = [
    "http://localhost:8080",    # Frontend production server
    "http://localhost:5173",    # Frontend development server
    "http://localhost:11434",   # Ollama server
]


def _config_healthcheck() -> None:
    """Ensure config_updated.yaml exists and vector DB path is valid."""
    src: Path = CONFIG_PATH.parent.joinpath("config.yaml")
    if not CONFIG_PATH.exists(follow_symlinks=True):
        copyfile(
            src=src,
            dst=CONFIG_PATH,
            follow_symlinks=True
        )

    with CONFIG_PATH.open(
        mode="r",
        encoding="utf-8"
    ) as config_file:
        config: dict[str, Any] = safe_load(config_file)

    if not Path(config["rag"]["vector_db_path"]).resolve(strict=True).exists(follow_symlinks=True):
        config["rag"]["vector_db_path"] = "data/vector_db_cosmic"

    with CONFIG_PATH.open(
        mode="w",
        encoding="utf-8"
    ) as config_file:
        safe_dump(
            data=config,
            stream=config_file,
            encoding="utf-8",
            sort_keys=False # Respect key-value order in oroginal config YAML file
        )

    return None


def _load_openai_key() -> str | None:
    api_key: dict[str, Any] = dotenv_values(
        dotenv_path=ENV_PATH,
        encoding= "utf-8"
    )

    return api_key.get(
        "OPENAI_API_KEY",
        None
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Equivalent to the explicit 'startup' event
    _config_healthcheck()

    openai_api_key: str | None = _load_openai_key()

    if openai_api_key is None:
        print(set_color("warning", "OPENAI_API_KEY is required in .env or environment variables."))
    else:
        cosmic_instance: OpenSICoSMIC = OpenSICoSMIC(
            config_path=str(CONFIG_PATH)
        )

        # Ref: https://www.starlette.dev/applications/#storing-state-on-the-app-instance
        app.state.opensi_cosmic             = cosmic_instance
        app.state.openai_api_key            = openai_api_key
        app.state.openai_api_status         = cosmic_instance.check_openai_key()
        app.state.config_path               = CONFIG_PATH
        app.state.config_modify_timestamp   = CONFIG_PATH.stat().st_mtime

    # Server is running
    yield

    # Equivalent to the explicit 'shutdown' event
    app.state.opensi_cosmic.quit()



# =====================Debugging========================
# Uncomment the following lines to enable debugging
# import debugpy
# print("Waiting for debugger attach...")
# debugpy.listen(("0.0.0.0", 5678))
# debugpy.wait_for_client()
# print("Debugger attached!")
# =======================================================


app: FastAPI = FastAPI(lifespan=lifespan)

# Allow CORS for the specified origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGIN, # or ["*"] to allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# APIs - smanile
app.include_router(router=default_apis.router)
app.include_router(router=models.router)
app.include_router(router=cosmic.router)
