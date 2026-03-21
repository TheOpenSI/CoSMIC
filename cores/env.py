from dotenv import dotenv_values
from pathlib import Path


def get_env() -> dict[str, str | None]:
    env_file = (Path(__file__).resolve(strict=True).parent / "fastapi.env")

    return dotenv_values(
        dotenv_path=env_file,
        encoding= "utf-8"
    )
