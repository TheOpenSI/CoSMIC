import os
import pandas as pd

statistic_dir = "data/cosmic/statistic"
statistic_dict = {
            "user_id": "unknown",
            "email": "unknown",
            "start_date": -1,
            "last_date": -1,
            "average_token_length": 0,
            "query_count": 0
        }

def update_statistic_table(statistic_dict):
    global statistic_dir

    os.makedirs(statistic_dir, exist_ok=True)
    current_time = statistic_dict["last_date"]
    time_split = current_time.split(",")[0].split("-")
    current_month_year = f"{time_split[1]}-{time_split[2]}"
    statistic_path = os.path.join(
        statistic_dir,
        f"{current_month_year}.csv"
    )

    if os.path.exists(statistic_path):
        data = pd.read_csv(statistic_path)
        user_emails = data["email"].tolist()
    else:
        user_emails = []

    if statistic_dict["email"] in user_emails:
        idx = [idx for idx, user_email in enumerate(user_emails) if user_email == statistic_dict["email"]][0]
        data.loc[idx, "last_date"] = statistic_dict["last_date"]
        history_total_token_length = data["average_token_length"][idx] * data["query_count"][idx]
        current_total_token_length = statistic_dict["average_token_length"] * statistic_dict["query_count"]
        total_query_count = data["query_count"][idx] + statistic_dict["query_count"]
        # TODO: I don't fucking know how this changes fix it, but it fucking works b4 we re-write this shit
        data.loc[idx, "average_token_length"] = int((history_total_token_length + current_total_token_length) / total_query_count)
        data.loc[idx, "query_count"] = total_query_count
    else:
        df = pd.DataFrame([{
            "user_id": statistic_dict["user_id"],
            "email": statistic_dict["email"],
            "start_date": statistic_dict["start_date"],
            "last_date": statistic_dict["last_date"],
            "average_token_length": statistic_dict["average_token_length"],
            "query_count": statistic_dict["query_count"]
        }])
        if len(user_emails) > 0: df = pd.concat([data, df], axis=0)
        data = df

    data.to_csv(
        statistic_path,
        header=[
            "user_id",
            "email",
            "start_date",
            "last_date",
            "average_token_length",
            "query_count"
        ],
        index=False
    )

def update_statistic_per_query(
        query,
        user_id,
        user_email,
        current_time
    ):
        global statistic_dict
        if False:
            # Save for the previous user (when the user_id changed).
            pre_user_id = statistic_dict["user_id"]
            pre_query_count = statistic_dict["query_count"]
            token_length = len(query)

            # Accumulate for the same user.
            if pre_user_id == user_id:
                pre_average_token_length = statistic_dict["average_token_length"]
                statistic_dict["last_date"] = current_time
                statistic_dict["average_token_length"] = \
                    (pre_average_token_length * pre_query_count + token_length) \
                    / (pre_query_count + 1)
                statistic_dict["query_count"] = pre_query_count + 1

            if pre_user_id != user_id:
                # Save previous user statistic.
                if pre_user_id != "unknown":
                    update_statistic_table(statistic_dict)

                # Initialize for a different user.
                statistic_dict["user_id"] = user_id
                statistic_dict["email"] = user_email
                statistic_dict["start_date"] = current_time
                statistic_dict["last_date"] = current_time
                statistic_dict["average_token_length"] = token_length
                statistic_dict["query_count"] = 1
        else:
            # Save every query for the current user.
            token_length = len(query)
            statistic_dict["user_id"] = user_id
            statistic_dict["email"] = user_email
            statistic_dict["start_date"] = current_time
            statistic_dict["last_date"] = current_time
            statistic_dict["average_token_length"] = token_length
            statistic_dict["query_count"] = 1
            update_statistic_table(statistic_dict)
