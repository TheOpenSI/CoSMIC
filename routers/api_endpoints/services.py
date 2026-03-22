### Core modules ###
from fastapi import APIRouter, HTTPException, status
from sqlmodel import select


### Type hints ###
from uuid import UUID
from typing_extensions import Any, Sequence


### Internal modules ###
from ...cores.db import SessionDependency
from ...apis.models import (
    Services,
    ServiceCreate,
    ServicePublic,
    ServiceUpdate,
    ServiceDelete
)


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
    service: ServiceCreate,
    session: SessionDependency
) -> Any:
    service_db: Services = Services.model_validate(obj=service, strict=True)

    session.add(instance=service_db)
    session.commit()
    session.refresh(instance=service_db)

    return service_db


@services_v1_router.get(
    path="/{service_id}",
    status_code=status.HTTP_200_OK,
    response_model=ServicePublic
)
async def read_service_v1(
    service_id: UUID,
    session: SessionDependency
) -> Any:
    service_view: Services | None = session.get(entity=Services, ident=service_id)

    if service_view is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Service Not Found!"
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
            detail="Service Not Found!"
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
            detail="Service Not Found!"
        )
    else:
        session.delete(instance=service_gone)
        session.commit()

        return service_gone
