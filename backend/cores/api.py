# -------------------------------------------------------------------------------------------------------------
# File: api.py
# Project: Open Source Institute-Cognitive System of Machine Intelligent Computing (OpenSI-CoSMIC)
# Contributors:
#     Bing Tran <binhsan1307@gmail.com>
#
# Copyright (c) 2026 Open Source Institute
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without
# limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so, subject to the following
# conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial
# portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
# LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# -------------------------------------------------------------------------------------------------------------


### Core modules ###
from contextlib import asynccontextmanager
from fastapi import FastAPI


### Type hints ###


### Internal modules ###
from ..cores.db import create_db_and_table, create_default_account
from ..routers.normal_endpoints.root import root_router
from ..routers.normal_endpoints.home import home_router
# from ..routers.normal_endpoints.cosmic import cosmic_router
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
    create_default_account()
    yield

    # Equivalent to 'shutdown' event (Optional)


cosmic_app: FastAPI = FastAPI(lifespan=lifespan)


# Normal endpoints
cosmic_app.include_router(router=root_router)
cosmic_app.include_router(router=home_router)
# cosmic_app.include_router(router=cosmic_router)


# API endpoints (V1)
cosmic_app.include_router(router=users_v1_router)
cosmic_app.include_router(router=roles_v1_router)
cosmic_app.include_router(router=chatboxes_v1_router)
cosmic_app.include_router(router=services_v1_router)
cosmic_app.include_router(router=models_v1_router)
cosmic_app.include_router(router=statistics_v1_router)
