from sqlmodel import Field, SQLModel
from .cruds import *


class UserBase(SQLModel):
    """docstring for UserBase."""
    name: str = Field(index=True)
    email: str | None = Field(
        default=None,
        unique=True
    )
