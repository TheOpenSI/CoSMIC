from sqlalchemy.orm import Session
from datetime import datetime
from internal.models import Statistic
from zoneinfo import ZoneInfo

def update_statistic_table(query, user_id, user_email, db: Session):
    """
    Updates the statistics table in the database.
    """
    current_time = datetime.now(tz=ZoneInfo("Australia/Sydney"))
    token_length = len(query)
    statistic_dict = {
        "user_id": user_id,
        "email": user_email,
        "start_date": current_time,
        "last_date": current_time,
        "average_token_length": token_length,
        "query_count": 1
    }

    # Check if the user already exists in the statistics table
    statistic_entry = db.query(Statistic).filter_by(user_id=statistic_dict["user_id"]).first()

    if statistic_entry:
        # Update existing entry
        statistic_entry.last_date = statistic_dict["last_date"]
        history_total_token_length = statistic_entry.average_token_length * statistic_entry.query_count
        current_total_token_length = statistic_dict["average_token_length"] * statistic_dict["query_count"]
        total_query_count = statistic_entry.query_count + statistic_dict["query_count"]
        statistic_entry.average_token_length = (history_total_token_length + current_total_token_length) / total_query_count
        statistic_entry.query_count = total_query_count
    else:
        # Create a new entry
        new_statistic = Statistic(
            user_id=statistic_dict["user_id"],
            email=statistic_dict["email"],
            start_date=statistic_dict["start_date"],
            last_date=statistic_dict["last_date"],
            average_token_length=statistic_dict["average_token_length"],
            query_count=statistic_dict["query_count"]
        )
        db.add(new_statistic)

    db.commit()
