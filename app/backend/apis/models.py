import uuid
from sqlmodel import Field, SQLModel, text


class Users(SQLModel, table=True):
    # Postgres will generate the UUID (version 7) for us instead of manual defining it
    id: uuid.UUID | None = Field(
        # Python can ignore this safely
        default=None,
        primary_key=True,
        # Postgres can genereate UUID7 type based on this
        sa_column_kwargs={
            "server_default": text(text="uuidv7()"),
            "nullable": False
        }
    )
    # usr_role_id: int = Field(default=None, foreign_key=True)
    name: str
    email: str | None = Field(
        default=None,
        unique=True
    )
    pwd: str
