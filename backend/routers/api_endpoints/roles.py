### Core modules ###
from fastapi import APIRouter
from fastapi import HTTPException
from sqlmodel import select


### Type hints ###
from uuid import UUID
from sqlmodel.sql.expression import SelectOfScalar
from typing_extensions import Any, Sequence


### Internal modules ###
from ...cores.db import SessionDependency
from ...apis.cruds.roles import Roles, RoleCreate, RoleUpdate, RolePublic


router: APIRouter = APIRouter()


@router.get(
    path="/api/roles",
    tags=["API Endpoints"],
    response_model=list[RolePublic]
)
async def read_roles(
    session: SessionDependency
) -> Sequence[Roles]:
    roles_stmt: SelectOfScalar[Roles] = select(Roles)
    roles_view: Sequence[Roles] = session.exec(statement=roles_stmt).all()

    return roles_view


@router.post(
    path="/api/roles",
    tags=["API Endpoints"],
    response_model=RolePublic
)
async def create_roles(
    role: RoleCreate,
    session: SessionDependency
) -> Roles:
    role_db: Roles = Roles.model_validate(obj=role)

    session.add(instance=role_db)
    session.commit()
    session.refresh(role_db)

    return role_db


@router.get(
    path="/api/{role_id}",
    tags=["API Endpoints"],
    response_model=RolePublic
)
async def read_role(
    role_id: UUID,
    session: SessionDependency
) -> Roles:
    role_view: Roles | None = session.get(entity=Roles, ident=role_id)

    if role_view is None:
        raise HTTPException(
            status_code=404,
            detail="Role not found!"
        )
    else:
        return role_view


@router.patch(
    path="/api/{role_id}",
    tags=["API Endpoints"],
    response_model=RolePublic
)
async def update_role(
    role_id: UUID,
    role: RoleUpdate,
    session: SessionDependency
) -> Roles:
    role_db: Roles | None = session.get(entity=Roles, ident=role_id)

    if role_db is None:
        raise HTTPException(
            status_code=404,
            detail="Role not found!"
        )
    else:
        role_data: dict[str, Any] = role.model_dump(exclude_unset=True)
        role_db.sqlmodel_update(obj=role_data)

        session.add(instance=role_db)
        session.commit()
        session.refresh(instance=role_db)

        return role_db


@router.delete(
    path="/api/{role_id}",
    tags=["API Endpoints"]
)
async def delete_role(
    role_id: UUID,
    session: SessionDependency
) -> dict[str, str | bool]:
    role_data: Roles | None = session.get(entity=Roles, ident=role_id)

    if role_data is None:
        raise HTTPException(
            status_code=404,
            detail="Role not found!"
        )
    else:
        session.delete(instance=role_data)
        session.commit()

        return {
            "status": "OK"
        }
