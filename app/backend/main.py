### Core modules ###
from fastapi import FastAPI

### Internal modules ###
from .routers import home


app: FastAPI = FastAPI()


app.include_router(router=home.router)


@app.get(
    path="/",
    tags=["Default Endpoints"]
)
async def root() -> dict[str, str]:
    return {
        "message": "Welcome to CoSMIC!",
        "version": "0.1.0"
    }
