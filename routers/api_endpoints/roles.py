### Core modules ###
from fastapi import APIRouter, HTTPException, status
from sqlmodel import select


### Type hints ###
from uuid import UUID
from typing_extensions import Any, Sequence


### Internal modules ###
from ...cores.db import SessionDependency
from ...apis.models import (
    Roles,
    RoleCreate,
    RolePublic,
    RoleUpdate,
    RoleDelete
)


roles_v1_router: APIRouter = APIRouter(
    prefix="/v1/roles",
    tags=["Roles API (V1)"],
    responses={
        200: {
            "description": "OK (Roles API V1)"
        },
        201: {
            "description": "Created (Roles API V1)"
        },
        404: {
            "description": "Not Found (Roles API V1)"
        }
    }
)


@roles_v1_router.get(
    path="/",
    status_code=status.HTTP_200_OK,
    response_model=list[RolePublic]
)
async def read_roles_v1(
    session: SessionDependency
) -> Any:
    roles_view: Sequence[Roles] = session.exec(statement=select(Roles)).all()

    return roles_view


@roles_v1_router.post(
    path="/",
    status_code=status.HTTP_201_CREATED,
    response_model=RolePublic
)
async def create_role_v1(
    role: RoleCreate,
    session: SessionDependency
) -> Any:
    role_db: Roles = Roles.model_validate(obj=role, strict=True)

    session.add(instance=role_db)
    session.commit()
    session.refresh(instance=role_db)

    return role_db


@roles_v1_router.get(
    path="/{role_id}",
    status_code=status.HTTP_200_OK,
    response_model=RolePublic
)
async def read_role_v1(
    role_id: UUID,
    session: SessionDependency
) -> Any:
    role_view: Roles | None = session.get(entity=Roles, ident=role_id)

    if role_view is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Role Not Found!"
        )
    else:
        return role_view


@roles_v1_router.patch(
    path="/{role_id}",
    status_code=status.HTTP_200_OK,
    response_model=RolePublic
)
async def update_role_v1(
    role_id: UUID,
    role: RoleUpdate,
    session: SessionDependency
) -> Any:
    role_db: Roles | None = session.get(entity=Roles, ident=role_id)

    if role_db is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Role Not Found!"
        )
    else:
        role_data: dict[str, Any] = role.model_dump(exclude_unset=True)
        role_db.sqlmodel_update(obj=role_data)

        session.add(instance=role_db)
        session.commit()
        session.refresh(instance=role_db)

        return role_db


@roles_v1_router.delete(
    path="/{role_id}",
    status_code=status.HTTP_200_OK,
    response_model=RoleDelete
)
async def delete_role_v1(
    role_id: UUID,
    session: SessionDependency
) -> Any:
    role_gone: Roles | None = session.get(entity=Roles, ident=role_id)

    if role_gone is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Role Not Found!"
        )
    else:
        session.delete(instance=role_gone)
        session.commit()

        return role_gone
