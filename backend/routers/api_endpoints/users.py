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


# @router.get(
#     path="/api/users",
#     tags=["API Endpoints"],
#     response_model=list[UserPublic]
# )
# async def read_users(
#     session: SessionDependency
# ) -> Sequence[Users]:
#     users_stmt: SelectOfScalar[Users] = select(Users)
#     users_view: Sequence[Users] = session.exec(statement=users_stmt).all()
#
#     return users_view


@router.post(
    path="/api/user",
    tags=["API Endpoints"],
    # response_model=UserPublic
)
async def create_user(
    user: UserCreate,
    session: SessionDependency
) -> Users:
    # First 2 default roles for inital user and new user respectively
    role_admin: Roles = Roles(
        name="Admin",
        desc="Granted to initial user by default, update other users through this user on first time."
    )
    role_user: Roles = Roles(
        name="User",
        desc="Granted to new user by default."
    )

    # Create initial user that have all priviledges
    insert_initial_user: Users = Users(
        # TODO: read from .env instead
        name="cosmic",
        password="cosmic_password_123",
        granted=role_admin
    )
    # Create new sign up users
    new_user: Users = Users.model_validate(obj=user, strict=True)
    insert_new_user: Users = Users(
        name=new_user.name,
        email=new_user.email,
        password=new_user.password,
        granted=role_user
    )

    session.add(instance=insert_initial_user)
    session.add(instance=insert_new_user)

    session.commit()

    session.refresh(instance=insert_initial_user)
    session.refresh(instance=insert_new_user)

    return new_user


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
