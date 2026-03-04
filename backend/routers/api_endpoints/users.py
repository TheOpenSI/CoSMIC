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
from ...apis.cruds.users import Users, UserCreate, UserUpdate, UserPublic
from ...apis.cruds.roles import Roles


router: APIRouter = APIRouter()


@router.get(
    path="/api/users",
    tags=["API Endpoints"],
    response_model=list[UserPublic]
)
async def read_users(
    session: SessionDependency
) -> Sequence[Users]:
    users_stmt: SelectOfScalar[Users] = select(Users)
    users_view: Sequence[Users] = session.exec(statement=users_stmt).all()

    return users_view


@router.post(
    path="/api/user",
    tags=["API Endpoints"],
    response_model=UserPublic
)
async def create_user(
    user: UserCreate,
    session: SessionDependency
) -> Users:
    # Create user first
    user_data: dict[str, Any] = user.model_dump(
        exclude={
            "granted"
        }
    )
    user_db: Users = Users.model_validate(obj=user_data)

    # Create role associate with user second
    role_data: dict[str, Any] = user.role_dict.model_dump()
    role_db: Roles = Roles.model_validate(obj=role_data)
    role_db.assigned_to = user_db # Update the Relationship() data before commit

    # Create all together last
    session.add(instance=user_db)
    session.add(instance=role_db)

    session.commit()
    session.refresh(user_db)

    return user_db


@router.get(
    path="/api/{user_id}",
    tags=["API Endpoints"],
    response_model=UserPublic
)
async def read_user(
    user_id: UUID,
    session: SessionDependency
) -> Users:
    user_view: Users | None = session.get(entity=Users, ident=user_id)

    if user_view is None:
        raise HTTPException(
            status_code=404,
            detail="User not found!"
        )
    else:
        return user_view


@router.patch(
    path="/api/{user_id}",
    tags=["API Endpoints"],
    response_model=UserPublic
)
async def update_user(
    user_id: UUID,
    user: UserUpdate,
    session: SessionDependency
) -> Users:
    user_db: Users | None = session.get(entity=Users, ident=user_id)

    if user_db is None:
        raise HTTPException(
            status_code=404,
            detail="User not found!"
        )
    else:
        user_data: dict[str, Any] = user.model_dump(exclude_unset=True)
        user_db.sqlmodel_update(obj=user_data)

        session.add(instance=user_db)
        session.commit()
        session.refresh(instance=user_db)

        return user_db


@router.delete(
    path="/api/{user_id}",
    tags=["API Endpoints"]
)
async def delete_user(
    user_id: UUID,
    session: SessionDependency
) -> dict[str, str | bool]:
    user_data: Users | None = session.get(entity=Users, ident=user_id)

    if user_data is None:
        raise HTTPException(
            status_code=404,
            detail="User not found!"
        )
    else:
        session.delete(instance=user_data)
        session.commit()

        return {
            "status": "OK"
        }
