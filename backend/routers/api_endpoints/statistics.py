# -------------------------------------------------------------------------------------------------------------
# File: statistics.py
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
from ...apis.models import Statistics, StatisticCreate, StatisticPublic, StatisticPublicWithUser, StatisticUpdate, StatisticDelete


statistics_v1_router: APIRouter = APIRouter(
    prefix="/v1/statisitics",
    tags=["Statistics API (V1)"],
    responses={
        200: {
            "description": "OK (Statistics API V1)"
        },
        201: {
            "description": "Created (Statistics API V1)"
        },
        404: {
            "description": "Not Found (Statistics API V1)"
        }
    }
)


@statistics_v1_router.get(
    path="/",
    status_code=status.HTTP_200_OK,
    response_model=list[StatisticPublic]
)
async def read_statistics_v1(
    session: SessionDependency
) -> Any:
    statisitics_view: Sequence[Statistics] = session.exec(statement=select(Statistics)).all()

    return statisitics_view


@statistics_v1_router.post(
    path="/",
    status_code=status.HTTP_201_CREATED,
    response_model=StatisticPublic
)
async def create_statistic_v1(
    user_id: UUID,
    statistic: StatisticCreate,
    session: SessionDependency
) -> Any:
    statistic_db: Statistics = Statistics.model_validate(obj=statistic, strict=True)
    statistic_db.user_id = user_id

    session.add(instance=statistic_db)
    session.commit()
    session.refresh(instance=statistic_db)

    return statistic_db



@statistics_v1_router.get(
    path="/{statistic_id}",
    status_code=status.HTTP_200_OK,
    response_model=StatisticPublicWithUser
)
async def read_statistic_v1(
    statistic_id: UUID,
    session: SessionDependency
) -> Any:
    statistic_view: Statistics | None = session.get(entity=Statistics, ident=statistic_id)

    if statistic_view is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User Generated Statistic Not Found!"
        )
    else:
        return statistic_view


@statistics_v1_router.patch(
    path="/{statistic_id}",
    status_code=status.HTTP_200_OK,
    response_model=StatisticPublic
)
async def update_statistic_v1(
    statistic_id: UUID,
    statistic: StatisticUpdate,
    session: SessionDependency
) -> Any:
    statistic_db: Statistics | None = session.get(entity=Statistics, ident=statistic_id)

    if statistic_db is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User Generated Statistic Not Found!"
        )
    else:
        statistic_data: dict[str, Any] = statistic.model_dump(exclude_unset=True)
        statistic_db.sqlmodel_update(obj=statistic_data)

        session.add(instance=statistic_db)
        session.commit()
        session.refresh(instance=statistic_db)

        return statistic_db


@statistics_v1_router.delete(
    path="/{statistic_id}",
    status_code=status.HTTP_200_OK,
    response_model=StatisticDelete
)
async def delete_statistic_v1(
    statistic_id: UUID,
    session: SessionDependency
) -> Any:
    statistic_gone: Statistics | None = session.get(entity=Statistics, ident=statistic_id)

    if statistic_gone is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User Generated Statistic Not Found!"
        )
    else:
        session.delete(instance=statistic_gone)
        session.commit()

        return statistic_gone
