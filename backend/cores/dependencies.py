### Core modules ###
from pathlib import Path
from fastapi import Request


### Type hints ###


### Internal modules ###
from ...src.opensi_cosmic import OpenSICoSMIC



def get_opensi_cosmic(request: Request) -> OpenSICoSMIC:
    return request.app.state.opensi_cosmic


# TODO:
# these 3 funcs down here will get deleted for the same reason mentioned at the
# very start of `api.py` file
def get_openai_status(request: Request) -> str:
    return request.app.state.openai_api_status


def get_openai_key(request: Request) -> str:
    return request.app.state.openai_api_key


def get_config_path(request: Request) -> Path:
    return request.app.state.config_path
