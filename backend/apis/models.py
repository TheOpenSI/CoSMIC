### Core modules ###
from sqlmodel import Field, SQLModel, text
from sqlalchemy.schema import PrimaryKeyConstraint, UniqueConstraint, ForeignKeyConstraint


### Type hints ###


### Internal modules ###
from .cruds import *


class UserBase(SQLModel):
    """docstring for UserBase."""
    # This's the only way that SQLModel not trying to be a bitch for its "bridging" feature:
    # https://docs.sqlalchemy.org/en/21/orm/declarative_tables.html#orm-declarative-table-configuration
    __table_args__ = (
        PrimaryKeyConstraint(
            "id",
            name="PK_USER_ID"
        ),
        UniqueConstraint(
            "email",
            name="UK_USER_EMAIL"
        ),
    )

    name: str = Field(
        max_length=100,
        nullable=False
    )
    # Invoke `email` column is optional as by default, it's NULL in both
    # validation and database
    email: str | None = Field(
        default="",
        max_length=256,
        nullable=True,
        # Disable it unless you're fine with the default constraint configs
        # because SQLModel is shit.
        unique=False,
        sa_column_kwargs={
            "server_default": text(text="NULL")
        }
    )
    password: str = Field(
        max_length=256,
        nullable=False
    )


class RoleBase(SQLModel):
    """docstring for RoleBase."""
    # This's the only way that SQLModel not trying to be a bitch for its "bridging" feature:
    # https://docs.sqlalchemy.org/en/21/orm/declarative_tables.html#orm-declarative-table-configuration
    __table_args__ = (
        PrimaryKeyConstraint(
            "id",
            name="PK_ROLE_ID"
        ),
        ForeignKeyConstraint(
            columns=["user_id"],
            refcolumns=["users.id"],
            name="FK_ROLE_USER_ID",
            onupdate="CASCADE",
            ondelete="CASCADE",
            match="FULL"
        ),
        UniqueConstraint(
            "user_id",
            name="UK_ROLE_USER_ID"
        )
    )

    name: str = Field(
        max_length=20,
        nullable=False
    )
    # Invoke `desc` column is optional as by default, it's NULL in both
    # validation and database
    desc: str | None = Field(
        # TEXT type is effectively the same as VARCHAR with no length limitation
        default="",
        nullable=True,
        sa_column_kwargs={
            "server_default": text(text="NULL")
        }
    )
