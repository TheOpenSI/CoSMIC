from fastapi import APIRouter


router: APIRouter = APIRouter()


@router.get(
    path="/home",
    tags=["Regular Endpoints"]
)
async def home() -> dict[str, str]:
    return {
        "message": "This is the Home page of CoSMIC!"
    }
