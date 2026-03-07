### Core modules ###
from contextlib import asynccontextmanager
from fastapi import FastAPI


### Type hints ###


### Internal modules ###
from ..cores.db import create_db_and_table, create_default_account
from ..routers.normal_endpoints import root, home
from ..routers.api_endpoints import users, roles
from ..routers.api_endpoints.chatboxes import chatboxes_v1_router


# TODO: Need some more research on this usage rather than the deprecation
# event: 'startup' & 'shutdown'
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Equivalent to 'startup' event
    create_db_and_table()
    create_default_account()
    yield

    # Equivalent to 'shutdown' event (Optional)


cosmic_app: FastAPI = FastAPI(lifespan=lifespan)


# Non-API endpoints
cosmic_app.include_router(router=root.router)
cosmic_app.include_router(router=home.router)


# API endpoints
cosmic_app.include_router(router=users.router)
cosmic_app.include_router(router=roles.router)
cosmic_app.include_router(router=chatboxes_v1_router)
