### Core modules ###
from sqlmodel import (
    Field,
    SQLModel,
    Relationship,
    text
)
from sqlalchemy.schema import (
    PrimaryKeyConstraint,
    UniqueConstraint,
    ForeignKeyConstraint
)


### Type hints ###
from datetime import datetime, timezone
from uuid import UUID, uuid7
from sqlalchemy.types import (
    TIMESTAMP,
    Uuid,
    Text,
    VARCHAR
)
from sqlalchemy.dialects.postgresql import JSONB
from pydantic.networks import EmailStr
from pydantic.types import JsonValue
from typing import Optional


### Internal modules ###



################################################
### Base Model (inheritance by other models) ###
################################################
class UserBase(SQLModel):
    name: str = Field(
        max_length=100,
        nullable=False,
        sa_type=VARCHAR
    )


class RoleBase(SQLModel):
    name: str = Field(
        max_length=20,
        nullable=False,
        sa_type=VARCHAR
    )
    desc: str | None = Field(
        default=None,
        nullable=True,
        sa_type=Text
    )


class ChatboxBase(SQLModel):
    name: str = Field(
        max_length=256,
        nullable=False,
        sa_type=Text
    )
    details: dict[str, "JsonValue"] = Field(
        default={
            # Chat history format (show only on non-PATCH request due to its
            # overwrite features)
            "<user-role>": "<user-query>",
            "<model-name>": "<model-response>"
        },
        nullable=False,
        sa_type=JSONB
    )


class ServiceBase(SQLModel):
    name: str = Field(
        max_length=100,
        nullable=False,
        sa_type=VARCHAR
    )
    desc: str | None = Field(
        default=None,
        nullable=True,
        sa_type=Text
    )


class ModelBase(SQLModel):
    name: str = Field(
        max_length=100,
        nullable=False,
        sa_type=VARCHAR
    )
    provider: str = Field(
        max_length=100,
        nullable=False,
        sa_type=VARCHAR
    )
    desc: str | None = Field(
        default=None,
        nullable=True,
        sa_type=Text
    )


class StatisticBase(SQLModel):
    name: str | None = Field(
        default=None,
        nullable=True,
        sa_type=Text
    )
    details: dict[str, "JsonValue"] = Field(
        default={
            # Statistic page format (show only on non-PATCH request due to its
            # overwrite features)
            "<key>": "<value>"
        },
        nullable=False,
        sa_type=JSONB
    )



################################################
###   Table Link Models (Junction Tables)    ###
################################################
class UserRoleLink(SQLModel, table=True):
    """Junction table (Users + Roles)"""
    __tablename__: str = "user_roles" # type: ignore
    __table_args__: tuple[
        PrimaryKeyConstraint,
        ForeignKeyConstraint,
        ForeignKeyConstraint
    ] = (
        PrimaryKeyConstraint(
            "user_id", "role_id",
            name="PK_UR_ID"
        ),
        ForeignKeyConstraint(
            columns=["user_id"],
            refcolumns=["users.id"],
            name="FK_UR_USER_ID",
            onupdate="CASCADE",
            ondelete="CASCADE",
            match="FULL"
        ),
        ForeignKeyConstraint(
            columns=["role_id"],
            refcolumns=["roles.id"],
            name="FK_UR_ROLE_ID",
            onupdate="CASCADE",
            ondelete="CASCADE",
            match="FULL"
        ),
    )
    user_id: UUID | None = Field(
        default=None,
        nullable=False
    )
    role_id: UUID | None = Field(
        default=None,
        nullable=False
    )
    create_on: datetime | None = Field(
        default=None,
        nullable=False,
        sa_type=TIMESTAMP(timezone=True) # type: ignore
    )


class ServiceModelLink(SQLModel, table=True):
    """Junction table (Services + Models)"""
    __tablename__: str = "service_slms" # type: ignore
    __table_args__: tuple[
        PrimaryKeyConstraint,
        ForeignKeyConstraint,
        ForeignKeyConstraint
    ] = (
        PrimaryKeyConstraint(
            "service_id", "model_id",
            name="PK_SS_ID"
        ),
        ForeignKeyConstraint(
            columns=["service_id"],
            refcolumns=["services.id"],
            name="FK_SS_SERVICE_ID",
            onupdate="CASCADE",
            ondelete="CASCADE",
            match="FULL"
        ),
        ForeignKeyConstraint(
            columns=["model_id"],
            refcolumns=["models.id"],
            name="FK_SS_MODEL_ID",
            onupdate="CASCADE",
            ondelete="CASCADE",
            match="FULL"
        ),
    )
    service_id: UUID | None = Field(
        default=None,
        nullable=False
    )
    model_id: UUID | None = Field(
        default=None,
        nullable=False
    )
    create_on: datetime | None = Field(
        default=None,
        nullable=False,
        sa_type=TIMESTAMP(timezone=True) # type: ignore
    )


class UserChatSession(SQLModel, table=True):
    """Junction table (Users + Chatboxes + Services)"""
    __tablename__: str = "user_chat_sessions" # type: ignore
    __table_args__: tuple[
        PrimaryKeyConstraint,
        ForeignKeyConstraint,
        ForeignKeyConstraint,
        ForeignKeyConstraint
    ] = (
        PrimaryKeyConstraint(
            "user_id", "chatbox_id", "service_id",
            name="PK_UCS_ID"
        ),
        ForeignKeyConstraint(
            columns=["user_id"],
            refcolumns=["users.id"],
            name="FK_UCS_USER_ID",
            onupdate="CASCADE",
            ondelete="CASCADE",
            match="FULL"
        ),
        ForeignKeyConstraint(
            columns=["chatbox_id"],
            refcolumns=["chatboxes.id"],
            name="FK_UCS_CHATBOX_ID",
            onupdate="CASCADE",
            ondelete="CASCADE",
            match="FULL"
        ),
        ForeignKeyConstraint(
            columns=["service_id"],
            refcolumns=["services.id"],
            name="FK_UCS_SERVICE_ID",
            onupdate="CASCADE",
            ondelete="CASCADE",
            match="FULL"
        ),
    )
    user_id: UUID | None = Field(
        default=None,
        nullable=False
    )
    chatbox_id: UUID | None = Field(
        default=None,
        nullable=False
    )
    service_id: UUID | None = Field(
        default=None,
        nullable=False
    )
    status: bool = Field(
        default=False,
        sa_column_kwargs={
            "server_default": text(text="FALSE"),
            # Gotta explains the usecase of this column a bit
            "comment": "mainly for the multi-services usage scenario."
        }
    )
    create_on: datetime | None = Field(
        default=None,
        nullable=False,
        sa_type=TIMESTAMP(timezone=True) # type: ignore
    )



######################################################
### Table Model (dynamic database table creations) ###
######################################################
class Users(UserBase, table=True):
    __tablename__: str = "users" # type: ignore
    __table_args__: tuple[
        PrimaryKeyConstraint,
        UniqueConstraint
    ] = (
        PrimaryKeyConstraint(
            "id",
            name="PK_USER_ID"
        ),
        UniqueConstraint(
            "email",
            name="UK_USER_EMAIL"
        ),
    )
    email: str | None = Field(
        default=None,
        max_length=256,
        nullable=True,
        sa_type=VARCHAR
    )
    id: UUID = Field(
        default_factory=(lambda: uuid7()),
        primary_key=True,
        nullable=False,
        sa_type=Uuid
    )
    password: str = Field(
        max_length=256,
        nullable=False,
        sa_type=VARCHAR
    )
    create_on: datetime = Field(
        default_factory=(lambda: datetime.now(tz=timezone.utc)),
        nullable=False,
        sa_type=TIMESTAMP(timezone=True) # type: ignore
    )
    chatboxes: list["Chatboxes"] = Relationship(
        back_populates="users",
        cascade_delete=True,
        passive_deletes=True
    )
    statistics: Optional["Statistics"] = Relationship(
        back_populates="users",
        cascade_delete=True,
        passive_deletes=True
    )
    roles: list["Roles"] = Relationship(
        back_populates="users",
        link_model=UserRoleLink
    )
    chat_sessions: list["UserChatSession"] = Relationship()


class Roles(RoleBase, table=True):
    __tablename__: str = "roles" # type: ignore
    __table_args__: tuple[
        PrimaryKeyConstraint
    ]= (
        PrimaryKeyConstraint(
            "id",
            name="PK_ROLE_ID"
        ),
    )
    id: UUID = Field(
        default_factory=(lambda: uuid7()),
        primary_key=True,
        nullable=False,
        sa_type=Uuid
    )
    create_on: datetime = Field(
        default_factory=(lambda: datetime.now(tz=timezone.utc)),
        nullable=False,
        sa_type=TIMESTAMP(timezone=True) # type: ignore
    )
    users: list["Users"] = Relationship(
        back_populates="roles",
        link_model=UserRoleLink
    )


class Chatboxes(ChatboxBase, table=True):
    __tablename__: str = "chatboxes" # type: ignore
    __table_args__: tuple[
        PrimaryKeyConstraint,
        ForeignKeyConstraint
    ] = (
        PrimaryKeyConstraint(
            "id",
            name="PK_CHATBOX_ID"
        ),
        ForeignKeyConstraint(
            columns=["user_id"],
            refcolumns=["users.id"],
            name="FK_CHATBOX_USER_ID",
            onupdate="CASCADE",
            ondelete="CASCADE",
            match="FULL"
        ),
    )
    id: UUID = Field(
        default_factory=(lambda: uuid7()),
        primary_key=True,
        nullable=False,
        sa_type=Uuid
    )
    user_id: UUID = Field(
        # Not generated by Python as required to be provided from the API endpoint
        default=None,
        primary_key=True,
        nullable=False,
        sa_type=Uuid
    )
    create_on: datetime = Field(
        default_factory=(lambda: datetime.now(tz=timezone.utc)),
        nullable=False,
        sa_type=TIMESTAMP(timezone=True) # type: ignore
    )
    users: Optional["Users"] = Relationship(
        back_populates="chatboxes"
    )
    chat_sessions: list["UserChatSession"] = Relationship()


class Services(ServiceBase, table=True):
    __tablename__: str = "services" # type: ignore
    __table_args__: tuple[
        PrimaryKeyConstraint
    ] = (
        PrimaryKeyConstraint(
            "id",
            name="PK_SERVICE_ID"
        ),
    )
    id: UUID = Field(
        default_factory=(lambda: uuid7()),
        primary_key=True,
        nullable=False,
        sa_type=Uuid
    )
    create_on: datetime = Field(
        default_factory=(lambda: datetime.now(tz=timezone.utc)),
        nullable=False,
        sa_type=TIMESTAMP(timezone=True) # type: ignore
    )
    models: list["Models"] = Relationship(
        back_populates="services",
        link_model=ServiceModelLink
    )
    chat_sessions: list["UserChatSession"] = Relationship()


class Models(ModelBase, table=True):
    __tablename__: str = "models" # type: ignore
    __table_args__: tuple[
        PrimaryKeyConstraint
    ] = (
        PrimaryKeyConstraint(
            "id",
            name="PK_MODEL_ID"
        ),
    )
    id: UUID = Field(
        default_factory=(lambda: uuid7()),
        primary_key=True,
        nullable=False,
        sa_type=Uuid
    )
    install_on: datetime | None = Field(
        default_factory=(lambda: datetime.now(tz=timezone.utc)),
        nullable=False,
        sa_type=TIMESTAMP(timezone=True) # type: ignore
    )
    services: list["Services"] = Relationship(
        back_populates="models",
        link_model=ServiceModelLink
    )


class Statistics(StatisticBase, table=True):
    __tablename__: str = "statistics" # type: ignore
    __table_args__: tuple[
        PrimaryKeyConstraint,
        ForeignKeyConstraint
    ] = (
        PrimaryKeyConstraint(
            "id",
            name="PK_STATISTIC_ID"
        ),
        ForeignKeyConstraint(
            columns=["user_id"],
            refcolumns=["users.id"],
            name="FK_STATISTIC_USER_ID",
            onupdate="CASCADE",
            ondelete="CASCADE"
        ),
    )
    id: UUID = Field(
        default_factory=(lambda: uuid7()),
        primary_key=True,
        nullable=False,
        sa_type=Uuid
    )
    user_id: UUID = Field(
        # Not generated by Python as required to be provided from the API endpoint
        default=None,
        primary_key=True,
        nullable=False,
        sa_type=Uuid
    )
    create_on: datetime | None = Field(
        default_factory=(lambda: datetime.now(tz=timezone.utc)),
        nullable=False,
        sa_type=TIMESTAMP(timezone=True) # type: ignore
    )
    users: Optional["Users"] = Relationship(
        back_populates="statistics",
        sa_relationship_kwargs={
            # This allows delete operation on the FK side of 1-1
            "single_parent": True
        },
        cascade_delete=True
    )



#####################################################
### Data Model (dynamic database data operations) ###
#####################################################
class UserPublic(UserBase):
    email: EmailStr | None = None
    id: UUID
    # TODO: implement hashed password
    password: str
    create_on: datetime


class UserCreate(UserBase):
    # TODO: implement hashed password
    email: EmailStr | None = None
    password: str


class UserUpdate(UserBase):
    name:       str | None = None # type: ignore
    email:      EmailStr | None = None
    # TODO: implement hashed password
    password:   str | None = None # type: ignore


class UserDelete(UserBase):
    id: UUID
    create_on: datetime
    response: dict[str, int | str] = {
        "status": 200,
        "message": "OK"
    }


class RolePublic(RoleBase):
    id: UUID
    create_on: datetime


class RoleCreate(RoleBase):
    pass


class RoleUpdate(RoleBase):
    name:   str | None = None # type: ignore
    desc:   str | None = None


class RoleDelete(RoleBase):
    id: UUID
    create_on: datetime
    response: dict[str, int | str] = {
        "status": 200,
        "message": "OK"
    }


class ChatboxPublic(ChatboxBase):
    # Use mainly for POST & PATCH requests as no need to view full FK data
    id: UUID
    user_id: UUID
    create_on: datetime


class ChatboxPublicWithUser(ChatboxBase):
    # No need to see user ID here as it's already in the Relationship()
    id: UUID
    users: UserPublic | None = None
    create_on: datetime


class ChatboxCreate(ChatboxBase):
    pass


class ChatboxUpdate(ChatboxBase):
    name: str | None = None # type: ignore
    details: dict[str, "JsonValue"] | None = { # type: ignore
        # Chat history format (as explained from the inheritance class, need to
        # define explicitly)
        "<user-role>": "<user-query>",
        "<model-name>": "<model-response>"
    }


class ChatboxDelete(ChatboxBase):
    id: UUID
    user_id: UUID
    create_on: datetime
    response: dict[str, int | str] = {
        "status": 200,
        "message": "OK"
    }


class ServicePublic(ServiceBase):
    id: UUID
    create_on: datetime


class ServiceCreate(ServiceBase):
    pass


class ServiceUpdate(ServiceBase):
    name:   str | None = None # type: ignore
    desc:   str | None = None


class ServiceDelete(ServiceBase):
    id: UUID
    create_on: datetime
    response: dict[str, int | str] = {
        "status": 200,
        "message": "OK"
    }


class ModelPublic(ModelBase):
    id: UUID
    install_on: datetime


class ModelCreate(ModelBase):
    pass


class ModelUpdate(ModelBase):
    name:       str | None = None # type: ignore
    provider:   str | None = None # type: ignore
    desc:       str | None = None


class ModelDelete(ModelBase):
    id: UUID
    install_on: datetime
    response: dict[str, int | str] = {
        "status": 200,
        "message": "OK"
    }


class StatisticPublic(StatisticBase):
    id: UUID
    user_id: UUID
    create_on: datetime


class StatisticPublicWithUser(StatisticBase):
    # No need to see user ID here as it's already in the Relationship()
    id: UUID
    users: UserPublic | None = None
    create_on: datetime


class StatisticCreate(StatisticBase):
    pass


class StatisticUpdate(StatisticBase):
    name: str | None = None # type: ignore
    details: dict[str, "JsonValue"] | None = { # type: ignore
        # Statistic page format (as explained from the inheritance class, need to
        # define explicitly)
        "<key>": "<value>"
    }


class StatisticDelete(StatisticBase):
    id: UUID
    user_id: UUID
    create_on: datetime
    response: dict[str, int | str] = {
        "status": 200,
        "message": "OK"
    }
