### Core modules ###
from contextlib import asynccontextmanager
from os import environ
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


### Type hints ###
from typing import Any


### Internal modules ###
from .src.opensi_cosmic import OpenSICoSMIC
from .backend.routers import (
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



@asynccontextmanager
async def lifespan(app: FastAPI):
    # Equivalent to the explicit 'startup' event
    opensi_cosmic_instance: OpenSICoSMIC = OpenSICoSMIC()
    opensi_cosmic_default_configs: list[dict[str, dict[str, Any]]] = opensi_cosmic_instance.get_configs()

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
    app.state.default_configs = (
        # Similar trick to prevent having to perform expensive for-loop. Take a
        # look at `src/opensi_cosmic.py` [Line 85]
        list((opensi_cosmic_default_configs[0]).values())[0]
        if   (opensi_cosmic_default_configs)
        else ({})
    )

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

cosmic_app.include_router(models.router)
cosmic_app.include_router(cosmic.router)
cosmic_app.include_router(memory.router)
