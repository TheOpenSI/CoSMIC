### Core modules ###
from fastapi import APIRouter, status


### Type hints ###


### Internal modules ###


root_router: APIRouter = APIRouter(
    tags=["Root Endpoints"],
    responses={
        200: {
            "description": "OK (Root Endpoints)"
        },
        405: {
            "description": "Method Not Allowed (Root Endpoints)"
        }
    }
)


@root_router.get(
    path="/",
    status_code=status.HTTP_200_OK,
    response_model=None
)
async def read_root() -> dict[str, str]:
    return {
        "message": "Welcome to CoSMIC project!",
        "version": "0.1"
    }


@root_router.post(
    path="/",
    status_code=status.HTTP_405_METHOD_NOT_ALLOWED,
    response_model=None
)
async def create_root() -> dict[str, int | str]:
    return {
        "status": status.HTTP_405_METHOD_NOT_ALLOWED,
        "message": "POST request method is not allowed for Root Endpoints"
    }


@root_router.patch(
    path="/",
    status_code=status.HTTP_405_METHOD_NOT_ALLOWED,
    response_model=None
)
async def update_root() -> dict[str, int | str]:
    return {
        "status": status.HTTP_405_METHOD_NOT_ALLOWED,
        "message": "PATCH request method is not allowed for Root Endpoints"
    }


@root_router.delete(
    path="/",
    status_code=status.HTTP_405_METHOD_NOT_ALLOWED,
    response_model=None
)
async def delete_root() -> dict[str, int | str]:
    return {
        "status": status.HTTP_405_METHOD_NOT_ALLOWED,
        "message": "DELETE request method is not allowed for Root Endpoints"
    }
