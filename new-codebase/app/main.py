from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

# from .backend.api import <api-file>

FE_DIR: Path = Path("frontend/dist").resolve()


app: FastAPI = FastAPI(
    title="OpenSI-CoSMIC",
    version="1.0.0"
)

# Serve all built static files (HTML, CSS, JS, images, etc)
app.mount(
    path="/",
    app=StaticFiles(
        directory=FE_DIR,
        html=True
    ),
    name="frontend"
)


# app.include_router(router=<api-file>.router)

