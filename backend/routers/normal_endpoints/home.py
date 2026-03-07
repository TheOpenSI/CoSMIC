### Core modules ###
from fastapi import APIRouter, status


### Type hints ###


### Internal modules ###


home_router: APIRouter = APIRouter(
    prefix="/home",
    tags=["Home Endpoints"],
    responses={
        200: {
            "description": "OK (Home Endpoints)"
        },
        404: {
            "description": "Not Found (Home Endpoints)"
        },
        405: {
            "description": "Method Not Allowed (Home Endpoints)"
        }
    }
)


@home_router.get(
    path="/",
    status_code=status.HTTP_200_OK,
    response_model=None
)
async def read_home() -> str:
    return "This is the landing page of CoSMIC project!"


@home_router.post(
    path="/",
    status_code=status.HTTP_405_METHOD_NOT_ALLOWED,
    response_model=None
)
async def create_home() -> dict[str, int | str]:
    return {
        "status": 405,
        "message": "POST request method is not allowed for Home Endpoints"
    }


@home_router.patch(
    path="/",
    status_code=status.HTTP_405_METHOD_NOT_ALLOWED,
    response_model=None
)
async def update_home() -> dict[str, int | str]:
    return {
        "status": 405,
        "message": "PATCH request method is not allowed for Home Endpoints"
    }


@home_router.delete(
    path="/",
    status_code=status.HTTP_405_METHOD_NOT_ALLOWED,
    response_model=None
)
async def delete_home() -> dict[str, int | str]:
    return {
        "status": 405,
        "message": "DELETE request method is not allowed for Home Endpoints"
    }
