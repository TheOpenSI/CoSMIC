### Core modules ###
from fastapi import Request


### Type hints ###


### Internal modules ###
from ..src.opensi_cosmic import OpenSICoSMIC


def get_cosmic(request: Request) -> OpenSICoSMIC:
    return request.app.state.opensi_cosmic


def get_openai_status(request: Request) -> str:
    return request.app.state.openai_api_status


def get_openai_key(request: Request) -> str:
    return request.app.state.openai_api_key


def get_config_path(request: Request):
    return request.app.state.config_path
