### Core modules ###
from fastapi import APIRouter, HTTPException, status
from sqlmodel import select


### Type hints ###
from uuid import UUID
from typing_extensions import Any, Sequence


### Internal modules ###
from ...cores.db import SessionDependency
from ...apis.models import Chatboxes, ChatboxCreate, ChatboxUpdate, ChatboxPublicWithUser, ChatboxPublic, ChatboxDelete


chatboxes_v1_router: APIRouter = APIRouter(
    prefix="/v1/chatboxes",
    tags=["Chatboxes API (V1)"],
    responses={
        200: {
            "description": "OK (Chatboxes API V1)"
        },
        201: {
            "description": "Created (Chatboxes API V1)"
        },
        404: {
            "description": "Not Found (Chatboxes API V1)"
        }
    }
)


@chatboxes_v1_router.get(
    path="/",
    status_code=status.HTTP_200_OK,
    response_model=list[ChatboxPublic]
)
async def read_chatboxes_v1(
    session: SessionDependency
) -> Any:
    chatboxes_view: Sequence[Chatboxes] = session.exec(statement=select(Chatboxes)).all()

    return chatboxes_view


@chatboxes_v1_router.post(
    path="/",
    status_code=status.HTTP_201_CREATED,
    response_model=ChatboxPublic
)
async def create_chatbox_v1(
    user_id: UUID,
    chatbox: ChatboxCreate,
    session: SessionDependency
) -> Any:
    chatbox_db: Chatboxes = Chatboxes.model_validate(obj=chatbox, strict=True)
    chatbox_db.user_id = user_id

    session.add(instance=chatbox_db)
    session.commit()
    session.refresh(instance=chatbox_db)

    return chatbox_db



@chatboxes_v1_router.get(
    path="/{chatbox_id}",
    status_code=status.HTTP_200_OK,
    response_model=ChatboxPublicWithUser
)
async def read_chatbox_v1(
    chatbox_id: UUID,
    session: SessionDependency
) -> Any:
    chatbox_view: Chatboxes | None = session.get(entity=Chatboxes, ident=chatbox_id)

    if chatbox_view is None:
        raise HTTPException(
            status_code=404,
            detail="Provided chatbox that belonged to user not found!"
        )
    else:
        return chatbox_view


@chatboxes_v1_router.patch(
    path="/{chatbox_id}",
    status_code=status.HTTP_200_OK,
    response_model=ChatboxPublic
)
async def update_chatbox_v1(
    chatbox_id: UUID,
    chatbox: ChatboxUpdate,
    session: SessionDependency
) -> Any:
    chatbox_db: Chatboxes | None = session.get(entity=Chatboxes, ident=chatbox_id)

    if chatbox_db is None:
        raise HTTPException(
            status_code=404,
            detail="Chatbox Not Found."
        )
    else:
        chatbox_data: dict[str, Any] = chatbox.model_dump(exclude_unset=True)
        chatbox_db.sqlmodel_update(obj=chatbox_data)

        session.add(instance=chatbox_db)
        session.commit()
        session.refresh(instance=chatbox_db)

        return chatbox_db


@chatboxes_v1_router.delete(
    path="/{chatbox_id}",
    status_code=status.HTTP_200_OK,
    response_model=ChatboxDelete
)
async def delete_chatbox_v1(
    chatbox_id: UUID,
    session: SessionDependency
) -> Any:
    chatbox_gone: Chatboxes | None = session.get(entity=Chatboxes, ident=chatbox_id)

    if chatbox_gone is None:
        raise HTTPException(
            status_code=404,
            detail="Chatbox Not Found."
        )
    else:
        session.delete(instance=chatbox_gone)
        session.commit()

        return chatbox_gone
