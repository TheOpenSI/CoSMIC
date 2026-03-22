### Core modules ###
from contextlib import asynccontextmanager
from fastapi import FastAPI


### Type hints ###


### Internal modules ###
from ..cores.db import create_db_and_table

from ..routers.normal_endpoints.cosmic import cosmic_router

from ..routers.api_endpoints.users import users_v1_router
from ..routers.api_endpoints.roles import roles_v1_router
from ..routers.api_endpoints.chatboxes import chatboxes_v1_router
from ..routers.api_endpoints.services import services_v1_router
from ..routers.api_endpoints.models import models_v1_router
from ..routers.api_endpoints.statistics import statistics_v1_router


# TODO: Need some more research on this usage rather than the deprecation
# event: 'startup' & 'shutdown'
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Equivalent to 'startup' event
    create_db_and_table()

    yield

    # Equivalent to 'shutdown' event (Optional)


cosmic_app: FastAPI = FastAPI(lifespan=lifespan)


# Normal endpoints
cosmic_app.include_router(router=cosmic_router)


# API endpoints (V1)
cosmic_app.include_router(router=users_v1_router)
cosmic_app.include_router(router=roles_v1_router)
cosmic_app.include_router(router=chatboxes_v1_router)
cosmic_app.include_router(router=services_v1_router)
cosmic_app.include_router(router=models_v1_router)
cosmic_app.include_router(router=statistics_v1_router)
