from fastapi import FastAPI

# from .backend.api import <api-file>

app: FastAPI = FastAPI(
    title="OpenSI-CoSMIC",
    version="1.0.0"
)

# app.include_router(router=<api-file>.router)
