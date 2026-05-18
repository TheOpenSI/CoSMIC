### Core modules ###
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


### Type hints ###


### Internal modules ###
from .backend.cores.globals import (
    NEW_CONFIG_PATH,
    CORS_REQUEST_TIMEOUT,
    config_healthcheck,
    load_openai_key,
    CORS_ALLOW_ORIGIN,
    CORS_ALLOW_METHODS,
    CORS_ALLOW_HEADERS
)
from .backend.routers import (
    models,
    cosmic,
    default_apis
)
from .src.opensi_cosmic import OpenSICoSMIC
from .utils.log_tool import set_color


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Equivalent to the explicit 'startup' event
    config_healthcheck()

    openai_api_key:         str | None      = load_openai_key()
    opensi_cosmic_instance: OpenSICoSMIC    = OpenSICoSMIC(config_path=str(NEW_CONFIG_PATH))

    # Ref:
    # https://www.starlette.dev/applications/#storing-state-on-the-app-instance
    # https://fastapi.tiangolo.com/reference/fastapi/#fastapi.FastAPI.state
    app.state.opensi_cosmic             = opensi_cosmic_instance
    # TODO: these 4 will be deleted soon
    app.state.openai_api_key            = openai_api_key
    app.state.openai_api_status         = opensi_cosmic_instance.check_openai_key()
    app.state.config_path               = NEW_CONFIG_PATH
    app.state.config_modify_timestamp   = NEW_CONFIG_PATH.stat().st_mtime

    if openai_api_key is None:
        print(
            set_color(
                status="warning",
                information="OPENAI_API_KEY is required in .env or environment variables."
            )
        )

    else:
        # We're absolutely okay with not using OpenAI API key
        pass


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


cosmic_app: FastAPI = FastAPI(lifespan=lifespan)


# NOTE: enable CORS on application level (only on dev)
cosmic_app.add_middleware(
    middleware_class=CORSMiddleware,
    allow_credentials=True,
    allow_origins=CORS_ALLOW_ORIGIN,
    allow_methods=CORS_ALLOW_METHODS,
    allow_headers=CORS_ALLOW_HEADERS,
    max_age=CORS_REQUEST_TIMEOUT
)


# Normal endpoints
cosmic_app.include_router(router=default_apis.router)


# APIs endpoints (V1)
cosmic_app.include_router(router=models.router)
cosmic_app.include_router(router=cosmic.router)
