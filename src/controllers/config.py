from sqlalchemy.orm import Session
from internal.models import Config

def get_config(config_path: str, db: Session):
    """
    Fetch and return config from the database.

    Args:
        config_path (str): Path to the configuration file.
        db (Session): SQLAlchemy database session.

    Returns:
        Config: Config in the database.
    """
    try:
        config = db.query(Config).order_by(Config.id.desc()).first()
        if config:
            return config

    except Exception as e:
        print(f"Error fetching services: {e}")
        return []