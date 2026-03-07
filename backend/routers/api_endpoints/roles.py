### Core modules ###
from fastapi import APIRouter, HTTPException
from sqlmodel import select


### Type hints ###
from uuid import UUID
from typing_extensions import Any, Sequence


### Internal modules ###
from ...cores.db import SessionDependency
from ...apis.models import RoleDelete, Users, Roles, RoleUpdate, RolePublicWithUser, RolePublic


# NOTE:
# For our case, the relationship between `Roles` & `Users` table are 1-1.
# Therefore, we can't fetch roles directly without the user that has been
# assigned for the role. This, however, will create a bit of a confusion on why
# we specify the `{role_id}` path but not calling it anywhere. This's just the
# way that makes SQLModel playing nice with us based on reason above, as it
# using Pydantic to validate under the hood (which Pydantic don't know
# SQLAlchemy's Relationship API).


router: APIRouter = APIRouter()


@router.get(
    path="/api/{role_id}/roles",
    tags=["API Endpoints"],
    summary="Read All User Assigned Roles",
    response_model=list[RolePublic]
)
async def read_roles(
    session: SessionDependency
) -> Any:
    roles_view: Sequence[Roles] = session.exec(statement=select(Roles)).all()

    return roles_view


@router.get(
    path="/api/{user_id}/{role_id}",
    tags=["API Endpoints"],
    summary="Read User Assigned Role",
    response_model=RolePublicWithUser
)
async def read_role(
    role_id: UUID,
    session: SessionDependency
) -> Roles:
    role_view: Roles | None = session.get(entity=Roles, ident=role_id)

    if role_view is None:
        raise HTTPException(
            status_code=404,
            detail="User with assigned role not found!"
        )
    else:
        return role_view


@router.patch(
    path="/api/{user_id}/{role_id}",
    tags=["API Endpoints"],
    summary="Update User Assigned Role",
    response_model=RolePublic
)
async def update_role(
    role_id: UUID,
    role: RoleUpdate,
    session: SessionDependency
) -> Any:
    role_db: Roles | None = session.get(entity=Roles, ident=role_id)

    if role_db is None:
        raise HTTPException(
            status_code=404,
            detail="User with assigned role not found!"
        )
    else:
        role_data: dict[str, Any] = role.model_dump(exclude_unset=True)
        role_db.sqlmodel_update(obj=role_data)

        session.add(instance=role_db)
        session.commit()
        session.refresh(instance=role_db)

        return role_db


@router.delete(
    path="/api/{user_id}/{role_id}",
    tags=["API Endpoints"],
    summary="Delete User Assigned Role",
    response_model=RoleDelete
)
async def delete_role(
    role_id: UUID,
    session: SessionDependency
) -> Any:
    role_gone: Roles | None = session.get(entity=Roles, ident=role_id)

    if role_gone is None:
        raise HTTPException(
            status_code=404,
            detail="User with assigned role not found!"
        )
    else:
        session.delete(instance=role_gone)
        session.commit()

        return role_gone
