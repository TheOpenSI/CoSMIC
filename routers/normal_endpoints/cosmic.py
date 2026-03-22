### Core modules ###
from typing import Any
from fastapi import APIRouter, status


### Type hints ###


### Internal modules ###


cosmic_router: APIRouter = APIRouter(
    prefix="/cosmic",
    tags=["Cosmic Endpoints"],
    responses={
        200: {
            "description": "OK (Cosmic Endpoints)"
        },
        201: {
            "description": "Created (Cosmic Endpoints)"
        },
        404: {
            "description": "Not Found (Cosmic Endpoints)"
        },
        405: {
            "description": "Method Not Allowed (Cosmic Endpoints)"
        }
    }
)


@cosmic_router.get(
    path="/",
    status_code=status.HTTP_200_OK,
    response_model=None
)
async def read_cosmic() -> dict[str, str]:
    return {
        "message": "This is the cosmic page of CoSMIC project!"
    }


@cosmic_router.post(
    path="/",
    status_code=status.HTTP_201_CREATED,
    response_model=None
)
async def create_cosmic(
    payload: dict[str, Any]
) -> Any:
    # TODO:
    # Because we're doing this in a submodule approach, 1 way to resolve this's
    # to foward payload data from FE over to CoSMIC with a special port opened
    # (or through reverse proxy) and allowed by CORS. This's similar to how FE
    # sent data to us (e.g: localhost:5173/3000/cosmic).

    # {
    #   "user_message": "<user-query>",
    #   "body": {
    #     "user": {
    #       "id": "<UUID-v7>",
    #       "role": "<user-role>",
    #       "email": "<user-email>"
    #     },
    #     "messages":{
    #         "<user-role>": "<user-query>",
    #         "<CoSMIC-role>": "<CoSMIC-response>"
    #     }
    #   }
    # }
    return {
        "tag": "experimental",
        "desc": "Not yet implemented. Please try again later!",
        "data": payload
    }


@cosmic_router.patch(
    path="/",
    status_code=status.HTTP_405_METHOD_NOT_ALLOWED,
    response_model=None
)
async def update_cosmic() -> dict[str, int | str]:
    return {
        "status": status.HTTP_405_METHOD_NOT_ALLOWED,
        "message": "PATCH request method is not allowed for Cosmic Endpoints"
    }


@cosmic_router.delete(
    path="/",
    status_code=status.HTTP_405_METHOD_NOT_ALLOWED,
    response_model=None
)
async def delete_cosmic() -> dict[str, int | str]:
    return {
        "status": status.HTTP_405_METHOD_NOT_ALLOWED,
        "message": "DELETE request method is not allowed for Cosmic Endpoints"
    }
