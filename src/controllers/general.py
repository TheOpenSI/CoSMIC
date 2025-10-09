from sqlalchemy.orm import Session
from internal.models import Service

def get_all_services(db: Session):
    """
    Fetch and return all services from the database.

    Args:
        db (Session): SQLAlchemy database session.

    Returns:
        List[Service]: List of all services in the database.
    """
    try:
        services = db.query(Service).all()
        return services
    except Exception as e:
        print(f"Error fetching services: {e}")
        return []