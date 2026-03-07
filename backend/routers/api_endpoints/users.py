### Core modules ###
from fastapi import APIRouter, HTTPException
from sqlmodel import select


### Type hints ###
from uuid import UUID
from typing_extensions import Any, Sequence


### Internal modules ###
from ...cores.db import SessionDependency
from ...apis.models import Roles, Users, UserCreate, UserUpdate, UserPublicWithRole, UserPublic, UserDelete


router: APIRouter = APIRouter()


@router.get(
    path="/api/users",
    tags=["API Endpoints"],
    summary="Read All Users",
    response_model=list[UserPublic]
)
async def read_users(
    session: SessionDependency
) -> Any:
    users_view: Sequence[Users] = session.exec(statement=select(Users)).all()

    return users_view


@router.post(
    path="/api/user",
    tags=["API Endpoints"],
    response_model=UserPublic
)
async def create_user(
    user: UserCreate,
    session: SessionDependency
) -> Any:
    # New users have a default user role granted for them
    new_account: Users = Users(
        name=user.name,
        # Trick to handle NULL value (when needed) as Swagger gives literal
        # "string" value for optional data by default
        email=user.email if user.email != "string" else None,
        password=user.password,
        granted= Roles(
            name="User",
            desc=""
        )
    )

    # Final validation to ensure corect data type & value sent to database model
    user_db: Users = Users.model_validate(obj=new_account, strict=True)

    session.add(instance=user_db)
    session.commit()
    session.refresh(instance=user_db)

    return user_db


@router.get(
    path="/api/{user_id}",
    tags=["API Endpoints"],
    response_model=UserPublicWithRole
)
async def read_user(
    user_id: UUID,
    session: SessionDependency
) -> Any:
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
) -> Any:
    user_db: Users | None = session.get(entity=Users, ident=user_id)

    if user_db is None:
        raise HTTPException(
            status_code=404,
            detail="User not found!"
        )
    else:
        update_account: dict[str, Any] = user.model_dump(exclude_unset=True)
        user_db.sqlmodel_update(obj=update_account)

        session.add(instance=user_db)
        session.commit()
        session.refresh(instance=user_db)

        return user_db


@router.delete(
    path="/api/{user_id}",
    tags=["API Endpoints"],
    response_model=UserDelete
)
async def delete_user(
    user_id: UUID,
    session: SessionDependency
) -> Any:
    user_gone: Users | None = session.get(entity=Users, ident=user_id)

    if user_gone is None:
        raise HTTPException(
            status_code=404,
            detail="User not found!"
        )
    else:
        session.delete(instance=user_gone)
        session.commit()

        return user_gone
