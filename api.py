### Core modules ###
from os import environ
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

### Type hints ###


### Internal modules ###
from .backend.routers import default_apis, models, cosmic, memory
# from .backend.routers import default_apis, models, cosmic, chat_session_memory

# =====================Debugging========================
# Uncomment the following lines to enable debugging
# import debugpy
# print("Waiting for debugger attach...")
# debugpy.listen(("0.0.0.0", 5678))
# debugpy.wait_for_client()
# print("Debugger attached!")
# =======================================================


app = FastAPI()

# To test CORS_ALLOW_ORIGIN locally, you can set something like
# CORS_ALLOW_ORIGIN=http://localhost:5173;http://localhost:8080
# in your .env file depending on your frontend port, 8080 or 5173 in this case.

CORS_ALLOW_ORIGIN = [
    "http://localhost:8080",  # Frontend production server
    "http://localhost:5173",  # Frontend development server
    "http://localhost:11434",  # Ollama server
]

# Allow CORS for the specified origins
if environ.get("CORS_ALLOW_ORIGIN"):
    CORS_ALLOW_ORIGIN.extend(str(object=environ.get("CORS_ALLOW_ORIGIN")).split(";"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGIN,  # or ["*"] to allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# APIs - smanile
app.include_router(default_apis.router, tags=["CoSMIC APIs"])
app.include_router(models.router, prefix="/api/v1/models", tags=["Ollama Models APIs"])
app.include_router(cosmic.router, prefix="/api/v1/cosmic", tags=["CoSMIC - V1"])
app.include_router(
    memory.router,
    prefix="/api/v1/memory",
    tags=["Upload Files"],
)
