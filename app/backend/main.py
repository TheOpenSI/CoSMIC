### Core modules ###
from fastapi import FastAPI


### Internal modules ###
from .routers import home


cosmic_app: FastAPI = FastAPI()


cosmic_app.include_router(router=home.router)


@cosmic_app.get(
    path="/",
    tags=["Default Endpoints"]
)
async def root() -> dict[str, str]:
    return {
        "message": "Welcome to CoSMIC!",
        "version": "0.1.0"
    }
