### Core modules ###
from fastapi import APIRouter, HTTPException
from sqlmodel import select


### Type hints ###
from uuid import UUID
from typing_extensions import Any, Sequence


### Internal modules ###
from ...cores.db import SessionDependency
from ...apis.models import Users, Roles, RoleCreate, RoleUpdate, RolePublicWithUser, RolePublic


router: APIRouter = APIRouter()


@router.get(
    path="/api/roles",
    tags=["API Endpoints"],
    response_model=list[RolePublic]
)
async def read_roles(
    session: SessionDependency
) -> Any:
    roles_view: Sequence[Roles] = session.exec(statement=select(Roles)).all()

    return roles_view


# @router.post(
#     path="/api/{user_id}/role",
#     tags=["API Endpoints"],
#     response_model=RolePublic
# )
# async def create_role(
#     user_id: UUID,
#     role: RoleCreate,
#     session: SessionDependency
# ) -> Roles:
#     user_db: Users | None = session.get(entity=Users, ident=user_id)
#
#     if user_db is None:
#         raise HTTPException(
#             status_code=404,
#             detail="User not found!"
#         )
#
#     elif user_db.granted:
#         raise HTTPException(
#             status_code=400,
#             detail="User already have a role!"
#         )
#     else:
#         role_db: Roles = Roles.model_validate(obj=role)
#
#         session.add(instance=role_db)
#         session.commit()
#         session.refresh(role_db)
#
#         return role_db


@router.get(
    path="/api/{role_id}",
    tags=["API Endpoints"],
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
            detail="Role not found!"
        )
    else:
        return role_view


@router.patch(
    path="/api/{user_id}/role",
    tags=["API Endpoints"],
    response_model=RolePublic
)
async def update_role(
    user_id: UUID,
    role: RoleUpdate,
    session: SessionDependency
) -> Roles:
    user_db: Users | None = session.get(entity=Users, ident=user_id)

    if user_db is None:
        raise HTTPException(
            status_code=404,
            detail="User not found!"
        )

    elif user_db.granted is None:
        raise HTTPException(
            status_code=404,
            detail="User have no role yet!"
        )

    else:
        role_data: dict[str, Any] = role.model_dump(exclude_unset=True)
        user_db.granted.sqlmodel_update(obj=role_data)

        session.add(instance=user_db.granted)
        session.commit()
        session.refresh(instance=user_db.granted)

        return user_db.granted


@router.delete(
    path="/api/{user_id}/role",
    tags=["API Endpoints"]
)
async def delete_role(
    user_id: UUID,
    session: SessionDependency
) -> dict[str, str | bool]:
    user_db: Users | None = session.get(entity=Users, ident=user_id)

    if user_db is None:
        raise HTTPException(
            status_code=404,
            detail="User not found!"
        )

    elif user_db.granted is None:
        raise HTTPException(
            status_code=404,
            detail="User have no role yet!"
        )
    else:
        session.delete(instance=user_db.granted)
        session.commit()

        return {
            "status": True,
            "message": "Role is removed"
        }
