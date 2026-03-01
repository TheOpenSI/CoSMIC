### Core modules ###
from sqlmodel import Field, SQLModel
from sqlalchemy.schema import PrimaryKeyConstraint, UniqueConstraint


### Type hints ###


### Internal modules ###
from .cruds import *


class UserBase(SQLModel):
    """docstring for UserBase."""
    # One way of writing custom constraint name:
    # https://docs.sqlalchemy.org/en/21/orm/declarative_tables.html#orm-declarative-table-configuration
    __table_args__ = (
        PrimaryKeyConstraint("id", name="PK_USER_ID"),
        UniqueConstraint("email", name="UK_USER_EMAIL"),
    )

    name: str = Field(
        max_length=100
    )
    email: str | None = Field(
        default=None,
        max_length=256,
        unique=False # This's kinda funny but there's valid case for doing this
    )
    password: str = Field(
        max_length=256
    )
