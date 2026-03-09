# -------------------------------------------------------------------------------------------------------------
# File: chatboxes.py
# Project: Open Source Institute-Cognitive System of Machine Intelligent Computing (OpenSI-CoSMIC)
# Contributors:
#     Bing Tran <binhsan1307@gmail.com>
#
# Copyright (c) 2026 Open Source Institute
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without
# limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so, subject to the following
# conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial
# portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
# LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# -------------------------------------------------------------------------------------------------------------


### Core modules ###
from fastapi import APIRouter, HTTPException, status
from sqlmodel import select


### Type hints ###
from uuid import UUID
from typing_extensions import Any, Sequence


### Internal modules ###
from ...cores.db import SessionDependency
from ...apis.models import Chatboxes, ChatboxCreate, ChatboxPublic, ChatboxPublicWithUser, ChatboxUpdate, ChatboxDelete


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
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chatbox Not Found."
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
            status_code=status.HTTP_404_NOT_FOUND,
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
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chatbox Not Found."
        )
    else:
        session.delete(instance=chatbox_gone)
        session.commit()

        return chatbox_gone
