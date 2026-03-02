from fastapi import APIRouter


router: APIRouter = APIRouter()


@router.get(
    path="/",
    tags=["Normal Endpoints"]
)
async def root() -> dict[str, str]:
    return {
        "message": "Welcome to CoSMIC!",
        "version": "0.1.0"
    }
