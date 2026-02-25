### Core modules ###
from sqlmodel import SQLModel, Session, create_engine


### Type hints ###
from sqlalchemy.engine.base import Engine


### Custom modules ###
from .env import get_env
from ..apis import models


cosmic_db_configs = get_env()

cosmic_db_dia:  str | None = cosmic_db_configs["DB_DIALECT"]
cosmic_db_dri:  str | None = cosmic_db_configs["DB_DRIVER"]
cosmic_db_usr:  str | None = cosmic_db_configs["DB_USER"]
cosmic_db_pwd:  str | None = cosmic_db_configs["DB_PASSWORD"]
cosmic_db_hst:  str | None = cosmic_db_configs["DB_HOST"]
cosmic_db_port: str | None = cosmic_db_configs["DB_PORT"]
cosmic_db_name: str | None = cosmic_db_configs["DB_NAME"]

cosmic_db_url: str = (
    "{0:s}+{1:s}://{2:s}:{3:s}@{4:s}:{5:s}/{6:s}".format(
        cosmic_db_dia,
        cosmic_db_dri,
        cosmic_db_usr,
        cosmic_db_pwd,
        cosmic_db_hst,
        cosmic_db_port,
        cosmic_db_name
    )
)

cosmic_db_engine: Engine = create_engine(
    url=cosmic_db_url,
    echo=True
)


def test_create_users() -> None:
    batch_users: list[models.Users] = [
        models.Users(name="Boryslavir", email="u3295557@uni.canberra.edu.au", pwd="123"),
        models.Users(name="DeepInDark", email=None, pwd="456"),
        models.Users(name="cosmic", email="opensi@canberra.edu.au", pwd="cosmic123")
    ]

    # Using `with` keyword, it handles starting/closing session automatically
    with Session(bind=cosmic_db_engine) as session_user:
        # Prepare data to commit
        session_user.add_all(instances=batch_users)

        # Execute and commit
        session_user.commit()

    return None


def main():
    # NOTE: 'echo' params are for debug only!
    SQLModel.metadata.create_all(bind=cosmic_db_engine)
    test_create_users()


if __name__ == "__main__":
    main()
