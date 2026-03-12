### Core modules ###
from fastapi import APIRouter, HTTPException, status
from sqlmodel import select


### Type hints ###
from uuid import UUID
from typing_extensions import Any, Sequence


### Internal modules ###
from ...cores.db import SessionDependency
from ...apis.models import Models, ModelCreate, ModelPublic, ModelPublicWithService, ModelUpdate, ModelDelete


models_v1_router: APIRouter = APIRouter(
    prefix="/v1/models",
    tags=["Models API (V1)"],
    responses={
        200: {
            "description": "OK (Models API V1)"
        },
        201: {
            "description": "Created (Models API V1)"
        },
        404: {
            "description": "Not Found (Models API V1)"
        }
    }
)


@models_v1_router.get(
    path="/",
    status_code=status.HTTP_200_OK,
    response_model=list[ModelPublic]
)
async def read_models_v1(
    session: SessionDependency
) -> Any:
    models_view: Sequence[Models] = session.exec(statement=select(Models)).all()

    return models_view


@models_v1_router.post(
    path="/",
    status_code=status.HTTP_201_CREATED,
    response_model=ModelPublic
)
async def create_model_v1(
    service_id: UUID,
    model: ModelCreate,
    session: SessionDependency
) -> Any:
    model_db: Models = Models.model_validate(obj=model, strict=True)
    model_db.service_id = service_id 

    session.add(instance=model_db)
    session.commit()
    session.refresh(instance=model_db)

    return model_db


@models_v1_router.get(
    path="/{model_id}",
    status_code=status.HTTP_200_OK,
    response_model=ModelPublicWithService
)
async def read_model_v1(
    model_id: UUID,
    session: SessionDependency
) -> Any:
    model_view: Models | None = session.get(entity=Models, ident=model_id)

    if model_view is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model Powered Service Not Found!"
        )
    else:
        return model_view


@models_v1_router.patch(
    path="/{model_id}",
    status_code=status.HTTP_200_OK,
    response_model=ModelPublic
)
async def update_model_v1(
    model_id: UUID,
    model: ModelUpdate,
    session: SessionDependency
) -> Any:
    model_db: Models | None = session.get(entity=Models, ident=model_id)

    if model_db is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model Powered Service Not Found!"
        )
    else:
        model_data: dict[str, Any] = model.model_dump(exclude_unset=True)
        model_db.sqlmodel_update(obj=model_data)

        session.add(instance=model_db)
        session.commit()
        session.refresh(instance=model_db)

        return model_db


@models_v1_router.delete(
    path="/{model_id}",
    status_code=status.HTTP_200_OK,
    response_model=ModelDelete
)
async def delete_model_v1(
    model_id: UUID,
    session: SessionDependency
) -> Any:
    model_gone: Models | None = session.get(entity=Models, ident=model_id)

    if model_gone is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model Powered Service Not Found!"
        )
    else:
        session.delete(instance=model_gone)
        session.commit()

        return model_gone
