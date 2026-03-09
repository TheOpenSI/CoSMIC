# -------------------------------------------------------------------------------------------------------------
# File: services.py
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
from fastapi import APIRouter, HTTPException, status
from sqlmodel import select


### Type hints ###
from uuid import UUID
from typing_extensions import Any, Sequence


### Internal modules ###
from ...cores.db import SessionDependency
from ...apis.models import Services, ServiceCreate, ServicePublic, ServicePublicWithChatbox, ServiceUpdate, ServiceDelete


services_v1_router: APIRouter = APIRouter(
    prefix="/v1/services",
    tags=["Services API (V1)"],
    responses={
        200: {
            "description": "OK (Services API V1)"
        },
        201: {
            "description": "Created (Services API V1)"
        },
        404: {
            "description": "Not Found (Services API V1)"
        }
    }
)


@services_v1_router.get(
    path="/",
    status_code=status.HTTP_200_OK,
    response_model=list[ServicePublic]
)
async def read_services_v1(
    session: SessionDependency
) -> Any:
    services_view: Sequence[Services] = session.exec(statement=select(Services)).all()

    return services_view


@services_v1_router.post(
    path="/",
    status_code=status.HTTP_201_CREATED,
    response_model=ServicePublic
)
async def create_service_v1(
    chatbox_id: UUID,
    service: ServiceCreate,
    session: SessionDependency
) -> Any:
    service_db: Services = Services.model_validate(obj=service, strict=True)
    service_db.chatbox_id = chatbox_id

    session.add(instance=service_db)
    session.commit()
    session.refresh(instance=service_db)

    return service_db


@services_v1_router.get(
    path="/{service_id}",
    status_code=status.HTTP_200_OK,
    response_model=ServicePublicWithChatbox
)
async def read_service_v1(
    service_id: UUID,
    session: SessionDependency
) -> Any:
    service_view: Services | None = session.get(entity=Services, ident=service_id)

    if service_view is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Service Not Found."
        )
    else:
        return service_view


@services_v1_router.patch(
    path="/{service_id}",
    status_code=status.HTTP_200_OK,
    response_model=ServicePublic
)
async def update_service_v1(
    service_id: UUID,
    service: ServiceUpdate,
    session: SessionDependency
) -> Any:
    service_db: Services | None = session.get(entity=Services, ident=service_id)

    if service_db is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Service Not Found."
        )
    else:
        service_data: dict[str, Any] = service.model_dump(exclude_unset=True)
        service_db.sqlmodel_update(obj=service_data)

        session.add(instance=service_db)
        session.commit()
        session.refresh(instance=service_db)

        return service_db


@services_v1_router.delete(
    path="/{service_id}",
    status_code=status.HTTP_200_OK,
    response_model=ServiceDelete
)
async def delete_service_v1(
    service_id: UUID,
    session: SessionDependency
) -> Any:
    service_gone: Services | None = session.get(entity=Services, ident=service_id)

    if service_gone is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Service Not Found."
        )
    else:
        session.delete(instance=service_gone)
        session.commit()

        return service_gone
