### Core modules ###
from sqlmodel import Field, SQLModel, text
from sqlalchemy.schema import PrimaryKeyConstraint, UniqueConstraint, ForeignKeyConstraint


### Type hints ###


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
