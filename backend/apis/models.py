### Core modules ###
from sqlmodel import Field, SQLModel, text
from sqlalchemy.schema import PrimaryKeyConstraint, UniqueConstraint, ForeignKeyConstraint


### Type hints ###
from uuid import UUID


### Internal modules ###


class UserBase(SQLModel):
    """docstring for UserBase."""
    # This's the only way that SQLModel not trying to be a bitch for its "bridging" feature:
    # https://docs.sqlalchemy.org/en/21/orm/declarative_tables.html#orm-declarative-table-configuration
    __table_args__ = (
        PrimaryKeyConstraint(
            "id",
            name="PK_USER_ID"
        ),
    )

    name: str = Field(
        max_length=100
    )
    email: str | None = Field(
        default=None,
        max_length=256,
        nullable=True,
        sa_column_kwargs={
            "server_default": text(text="NULL")
        }
    )
    password: str = Field(
        max_length=256
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
        max_length=20
    )
    desc: str | None = Field(
        # TEXT type is effectively the same as VARCHAR with no length limitation
        default=None,
        nullable=True,
        sa_column_kwargs={
            "server_default": text(text="NULL")
        }
    )
    # Postgres will generate the UUID (version 7) for us instead of manual
    # defining it. The way this works in SQLModel is to provide value to both
    # type-hint and `default` param
    user_id: UUID | None = Field(
        default=None,
        sa_column_kwargs={
            "server_default": text(text="uuidv7()")
        }
    )
