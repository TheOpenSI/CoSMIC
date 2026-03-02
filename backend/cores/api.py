### Core modules ###
from contextlib import asynccontextmanager
from uuid import UUID
from fastapi import FastAPI, HTTPException
from sqlmodel import select


### Type hints ###
from sqlmodel.sql.expression import SelectOfScalar
from typing_extensions import Any, Sequence


### Internal modules ###
from ..routers import home
from ..cores.db import SessionDependency, create_db_and_table
from ..apis.cruds.users import Users, UserCreate, UserUpdate, UserPublic


# TODO: Need some more research on this usage rather than the deprecation
# event: 'startup' & 'shutdown'
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Equivalent to 'startup' event
    create_db_and_table()
    yield

    # Equivalent to 'shutdown' event (Optional)


cosmic_app: FastAPI = FastAPI(lifespan=lifespan)


# Non-API endpoints
@cosmic_app.get(
    path="/",
    tags=["Normal Endpoints"]
)
async def root() -> dict[str, str]:
    return {
        "message": "Welcome to CoSMIC!",
        "version": "0.1.0"
    }

cosmic_app.include_router(router=home.router)


# API endpoints
@cosmic_app.get(
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

@cosmic_app.post(
    path="/api/users",
    tags=["API Endpoints"],
    response_model=UserPublic
)
async def create_users(
    user: UserCreate,
    session: SessionDependency
) -> Users:
    user_db: Users = Users.model_validate(obj=user)

    session.add(instance=user_db)
    session.commit()
    session.refresh(user_db)

    return user_db

@cosmic_app.get(
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

@cosmic_app.patch(
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

@cosmic_app.delete(
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
