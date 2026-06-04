### Core modules ###
from contextlib import asynccontextmanager
from os import environ
from pathlib import Path
from shutil import copyfile
from dotenv import dotenv_values
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from yaml import (
    safe_dump,
    safe_load
)


### Type hints ###
from typing import Any


### Internal modules ###
from .src.opensi_cosmic import OpenSICoSMIC
from .utils.log_tool import set_color
from .backend.routers import (
    default_apis,
    models,
    cosmic,
    memory
)



# =====================Debugging========================
# Uncomment the following lines to enable debugging:
#
# import debugpy
# print("Waiting for debugger attach...")
# debugpy.listen(("0.0.0.0", 5678))
# debugpy.wait_for_client()
# print("Debugger attached!")
# =======================================================



# TODO:
# these 3 global vars & 2 funcs will get deleted after we successfully replace
# `/config` endpoint in this repo by equivalent API endpoint from 'CoSMIC_DB' repo
OG_CONFIG_PATH:     Path = Path(__file__).resolve(strict=True).parent.joinpath(
    "scripts",
    "configs",
    "config.yaml"
)
NEW_CONFIG_PATH:    Path = OG_CONFIG_PATH.parent.joinpath("config_updated.yaml")
ENV_PATH:           Path = Path(__file__).resolve(strict=True).parent.joinpath(".env")


def config_healthcheck() -> None:
    if not NEW_CONFIG_PATH.exists(follow_symlinks=True):
        copyfile(
            src=OG_CONFIG_PATH,
            dst=NEW_CONFIG_PATH,
            follow_symlinks=True
        )
        return None

    else:
        with NEW_CONFIG_PATH.open(
            mode="r",
            encoding="utf-8"
        ) as config_file:
            config: dict[str, Any] = safe_load(stream=config_file)

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
    api_key: str | None = dotenv_values(
        dotenv_path=ENV_PATH,
        encoding="utf-8"
    ).get(
        "OPENAI_API_KEY",
        None
    )

    return api_key



@asynccontextmanager
async def lifespan(app: FastAPI):
    # Equivalent to the explicit 'startup' event

    # NOTE:
    # run healthcheck func first so `config_updated.yaml` is guaranteed to exist.
    # However, we'll soon remove this when the migration for `/config` endpoint
    # starts.
    config_healthcheck()

    opensi_cosmic_instance: OpenSICoSMIC = OpenSICoSMIC(config_path=str(NEW_CONFIG_PATH))

    # INFO:
    # ======================================================================== #
    # Let's say that during development, we did some changes to any file that
    # is/are Python file(s). Because hot-reload flag is enabled for FastAPI so
    # that we can see the changes immidiately, it'll always backtrack changes
    # starting from this file, then files that use `APIRouter()` class for
    # endpoints creation in `/docs` page, then finally file with changes.
    # Because of this, everytime `OpenSICoSMIC()` class get called/imported in
    # the first 2 checks, FastAPI will then have to re-initialise it twice so
    # that all other files can technically use methods defined in that class
    # without issues. However, this's the main issue that 'cause the double
    # initialisation issue, which could slow down development workflow depends
    # on your machine and installed model.
    #
    # By putting the core class into FastAPI's `State()`, it'll know that these
    # vars here needs to be loaded first before the API platform (or `/docs`
    # page) get started up. How does FastAPI know about this? That's just how it
    # works under the hood:
    #               [loaded any data into state during execution]
    #                                    |
    #                                    |
    #                                    v
    #                        [loaded API platform page]
    #                                    |
    #                                    |
    #                                    v
    #                            [finish execution]
    #
    # References:
    # [FastAPI docs](https://fastapi.tiangolo.com/reference/fastapi/#fastapi.FastAPI.state)
    # ======================================================================== #
    app.state.opensi_cosmic = opensi_cosmic_instance

    # TODO:
    # these logic down here will get deleted for the same reason mentioned at the
    # vert start of this file
    openai_api_key: str | None = load_openai_key()

    app.state.openai_api_key            = openai_api_key
    app.state.openai_api_status         = opensi_cosmic_instance.check_openai_key()
    app.state.config_path               = NEW_CONFIG_PATH
    app.state.config_modify_timestamp   = NEW_CONFIG_PATH.stat().st_mtime

    if openai_api_key is None:
        # No API key provided? No problem at all, we'll just use models pulled
        # from Ollama instead then
        print(
            set_color(
                status="warning",
                information="OPENAI_API_KEY is required in .env or environment variables."
            )
        )

    else:
        # Interesting, you sure want to spend your tokens on models from OpenAI?
        # Remember, open-sources and locals...
        pass


    # Server is running
    yield


    # Equivalent to the explicit 'shutdown' event
    app.state.opensi_cosmic.quit()


cosmic_app: FastAPI = FastAPI(lifespan=lifespan)


CORS_ALLOW_ORIGIN = [
    "http://localhost:8080",   # FE binded Docker port (prod)
    "http://localhost:5173",   # FE binded Docker port (dev)
    "http://localhost:11434",  # Ollama binded Docker port
]

# Allow CORS for the specified origins
if environ.get("CORS_ALLOW_ORIGIN"):
    CORS_ALLOW_ORIGIN.extend(str(object=environ.get("CORS_ALLOW_ORIGIN")).split(";"))

cosmic_app.add_middleware(
    middleware_class=CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGIN,  # or ["*"] to allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

cosmic_app.include_router(default_apis.router, tags=["CoSMIC APIs"])
cosmic_app.include_router(models.router, prefix="/api/v1/models", tags=["Ollama Models APIs"])
cosmic_app.include_router(cosmic.router, prefix="/api/v1/cosmic", tags=["CoSMIC - V1"])
cosmic_app.include_router(memory.router, prefix="/api/v1/memory", tags=["Upload Files"])
