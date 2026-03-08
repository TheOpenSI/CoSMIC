### Core modules ###
from fastapi import APIRouter, HTTPException, status
from sqlmodel import select


### Type hints ###
from uuid import UUID
from typing_extensions import Any, Sequence


### Internal modules ###
from ...cores.db import SessionDependency
from ...apis.models import Roles, Users, UserCreate, UserPublic, UserPublicWithRole, UserUpdate, UserDelete


users_v1_router: APIRouter = APIRouter(
    prefix="/v1/users",
    tags=["Users API (V1)"],
    responses={
        200: {
            "description": "OK (Users API V1)"
        },
        201: {
            "description": "Created (Users API V1)"
        },
        404: {
            "description": "Not Found (Users API V1)"
        }
    }
)


@users_v1_router.get(
    path="/",
    status_code=status.HTTP_200_OK,
    response_model=list[UserPublic]
)
async def read_users_v1(
    session: SessionDependency
) -> Any:
    users_view: Sequence[Users] = session.exec(statement=select(Users)).all()

    return users_view


@users_v1_router.post(
    path="/",
    status_code=status.HTTP_201_CREATED,
    response_model=UserPublic
)
async def create_user_v1(
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
        granted=Roles(
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


@users_v1_router.get(
    path="/{user_id}",
    status_code=status.HTTP_200_OK,
    response_model=UserPublicWithRole
)
async def read_user_v1(
    user_id: UUID,
    session: SessionDependency
) -> Any:
    user_view: Users | None = session.get(entity=Users, ident=user_id)

    if user_view is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User Not Found."
        )
    else:
        return user_view


@users_v1_router.patch(
    path="/{user_id}",
    status_code=status.HTTP_200_OK,
    response_model=UserPublic
)
async def update_user_v1(
    user_id: UUID,
    user: UserUpdate,
    session: SessionDependency
) -> Any:
    user_db: Users | None = session.get(entity=Users, ident=user_id)

    if user_db is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User Not Found."
        )
    else:
        user_data: dict[str, Any] = user.model_dump(exclude_unset=True)
        user_db.sqlmodel_update(obj=user_data)

        session.add(instance=user_db)
        session.commit()
        session.refresh(instance=user_db)

        return user_db


@users_v1_router.delete(
    path="/{user_id}",
    status_code=status.HTTP_200_OK,
    response_model=UserDelete
)
async def delete_user_v1(
    user_id: UUID,
    session: SessionDependency
) -> Any:
    user_gone: Users | None = session.get(entity=Users, ident=user_id)

    if user_gone is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User Not Found."
        )
    else:
        session.delete(instance=user_gone)
        session.commit()

        return user_gone
