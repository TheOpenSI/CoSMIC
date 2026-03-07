### Core modules ###
from sqlmodel import Field, SQLModel, Relationship, text
from sqlalchemy.schema import PrimaryKeyConstraint, UniqueConstraint, ForeignKeyConstraint


### Type hints ###
from datetime import datetime
from uuid import UUID
from sqlalchemy.sql.sqltypes import TIMESTAMP
from typing_extensions import Optional # SQLModel is being a bitch here, once again


### Internal modules ###



################################################
### Base Model (inheritance by other models) ###
################################################
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



######################################################
### Table Model (dynamic database table creations) ###
######################################################
class Roles(RoleBase, table=True):
    """docstring for Roles."""
    # Postgres will generate the UUID (version 7) for us instead of manual
    # defining it. The way this works in SQLModel is to provide value to both
    # type-hint and `default` param
    id: UUID | None = Field(
        default=None,
        # For some fucking reasons, SQLModel works perfectly fine with PK
        # constraint. For fuck sake!
        primary_key=True,
        sa_column_kwargs={
            "server_default": text(text="uuidv7()")
        }
    )
    # Postgres will generate the UUID (version 7) for us instead of manual
    # defining it. The way this works in SQLModel is to provide value to both
    # type-hint and `default` param
    user_id: UUID | None = Field(
        default=None,
        # U might asking where's the FK? Welp, go check `Base Model` section and
        # fucking find out why
        sa_column_kwargs={
            "server_default": text(text="uuidv7()")
        }
    )
    # Postgres will generate the timestamp (without time zone) for us instead of
    # manual defining it. The way this works in SQLModel is to provide value to
    # both type-hint and `default` param
    created_on: datetime | None = Field(
        default=None,
        nullable=False,
        sa_column_kwargs={
            "server_default": text(text="localtimestamp(2)")
        }
    )

    ### It's very confusing to read from database perspective, but here's my attempt:
    ###   "Admin role is assigned to CoSMIC user through `permission` object-level connection"
    ###     👆               👆        👆                    👆                        👆
    ###    Roles            Roles     Roles              Relationship             Relationship
    ###                     object  type-hint          `back_populates`
    assigned_to: Optional["Users"] = Relationship(
        back_populates="granted",
        sa_relationship_kwargs={
            "uselist": False,
            # This allows delete operation on the FK side of 1-1
            "single_parent": True
        },
        cascade_delete=True
    )


class Users(UserBase, table=True):
    """docstring for Users."""
    # TODO: implement hashed password
    password: str = Field(
        max_length=256
    )
    # Postgres will generate the UUID (version 7) for us instead of manual
    # defining it. The way this works in SQLModel is to provide value to both
    # type-hint and `default` param
    id: UUID | None = Field(
        default=None,
        nullable=False,
        # For some fucking reasons, SQLModel works perfectly fine with PK
        # constraint. For fuck sake!
        primary_key=True,
        sa_column_kwargs={
            "server_default": text(text="uuidv7()")
        }
    )
    # Postgres will generate the timestamp (with timezone) for us instead of
    # manual defining it. The way this works in SQLModel is to provide value to
    # both type-hint and `default` param
    created_on: datetime | None = Field(
        default=None,
        nullable=False,
        sa_type=TIMESTAMP(timezone=True), # type: ignore
        sa_column_kwargs={
            "server_default": text(text="current_timestamp(2)")
        }
    )

    ### It's very confusing to read from database perspective, but here's my attempt:
    ###   "CoSMIC user have Admin role granted through `assigned_to` object-level connection"
    ###      👆              👆          👆                 👆                        👆
    ###    Users            Users       Users           Relationship             Relationship
    ###                   type-hint    object         `back_populates`
    granted: Optional["Roles"] = Relationship(
        back_populates="assigned_to",
        sa_relationship_kwargs={
            "uselist": False
        },
        cascade_delete=True
    )



#####################################################
### Data Model (dynamic database data operations) ###
#####################################################

# NOTE:
# + This's why you don't just wrap over 2 popular libraries without having a
#   clear explanation on how to use them both in your library. Custom type-hints
#   from both of them, which are crucially needed for correct references, are
#   completely fucked.
#
# + Even the linter (pyright) is fucked with this kinda bridging as well. I
#   don't give a fuck anymore, I'll slap in `type: ignore` whenever I'm fucking
#   can.


class RolePublic(RoleBase):
    """docstring for RolePublic."""
    # Hint for testers that these columns will not be appear in body as it is
    # generated by database
    id: UUID
    user_id: UUID
    created_on: datetime


class RolePublicWithUser(RolePublic):
    """docstring for RolePublicWithUser."""
    # Hint for testers that these are Relationship Attributes (FK) value (GET
    # request scenario), which modifiable via the User API only
    assigned_to: UserPublic | None = None


class RoleUpdate(RoleBase):
    """docstring for RoleUpdate."""
    # Hint for testers that these columns are needed (optionally) alongside with
    # default one during a PATCH request
    name:       str | None = None # type: ignore
    desc:       str | None = None


class RoleDelete(RoleBase):
    """docstring for RoleDelete."""
    # Hint for testers that this is the response message for DELETE request
    id: UUID
    user_id: UUID
    created_on: datetime
    response: dict[str, int | str] | None = {
        "status": 200,
        "message": "User with assigned role deleted successfully."
    }


class UserPublic(UserBase):
    """docstring for UserPublic."""
    # Hint for testers that these columns will not be appear in body as it is
    # generated by database
    password: str # TODO: implement hashed password
    id: UUID
    created_on: datetime


class UserPublicWithRole(UserPublic):
    """docstring for UserPublicWithRole."""
    # Hint for testers that these are Relationship Attributes (FK) value (GET
    # request scenario), which modifiable via the Role API only
    granted: RolePublic | None = None


class UserCreate(UserBase):
    """docstring for UserCreate."""
    # Hint for testers that these columns are needed alongside with default one
    # during a POST request
    # TODO: implement hashed password
    password: str


class UserUpdate(UserBase):
    """docstring for UserUpdate."""
    # Hint for testers that these columns are needed (optionally) alongside with
    # default one during a PATCH request
    name:       str | None = None # type: ignore
    email:      str | None = None
    # TODO: implement hashed password
    password:   str | None = None # type: ignore


class UserDelete(UserBase):
    """docstring for UserDelete."""
    # Hint for testers that this is the response message for DELETE request
    id: UUID
    created_on: datetime
    response: dict[str, int | str] | None = {
        "status": 200,
        "message": "User deleted successfully."
    }
