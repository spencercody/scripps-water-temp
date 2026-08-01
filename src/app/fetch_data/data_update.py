from pathlib import Path

import pandas as pd

from app.fetch_data import fetch_daily_buoy_data, fetch_daily_weather, fetch_water_info

rel_path = Path("app_data/app-data.parquet")
cd = Path(__file__).resolve().parent.parent
DATA_PATH = cd / rel_path


def fetch_all_data():
    weather_df = fetch_daily_weather()
    water_df = fetch_water_info()
    wind_df = fetch_daily_buoy_data()

    merge_df = weather_df.merge(wind_df, how="left", on="date")

    full_df = merge_df.merge(water_df, how="left", on="date")

    return full_df


# ---------------------------------------------------------------------------- #


def fetch_existing_data(data_path=DATA_PATH):
    current_df = pd.read_parquet(data_path)
    current_df["date"] = pd.to_datetime(current_df.date)
    current_df['date'] = current_df.date.dt.normalize()
    current_df['date'] = current_df['date'].dt.tz_localize(None)
    return current_df


# ---------------------------------------------------------------------------- #


def update_data():

    # get existing data
    current_df = fetch_existing_data()

    # fetch new records
    incoming_records = fetch_all_data()

    # check if the data has changed from what already exists
    incoming_dates = set(incoming_records.date.unique())
    current_clean_df = current_df[current_df.date.isin(incoming_dates)]

    record_updates = 0
    for idx in current_clean_df.index:
        row = current_clean_df.loc[idx]

        # get the existing record
        current_record = row.to_dict()
        date = current_record["date"]

        # get the incoming record
        incoming_row = incoming_records[incoming_records.date == date]
        incoming_record_dic = incoming_row.to_dict(orient="records")[0]

        for key in current_record:
            incoming_value = incoming_record_dic[key]
            # if the current record value does not match the incoming
            if current_record[key] != incoming_value:
                # update the record tp the incoming value
                current_df.loc[idx, key] = incoming_value
                record_updates += 1

    # check for new records only
    most_recent_date = current_df.date.max()
    new_records_df = incoming_records[incoming_records.date > most_recent_date]
    n_new_records = len(new_records_df)
    if not new_records_df.empty:

        current_df = pd.concat([current_df, new_records_df], ignore_index=True)

    # ensure there are no duplicate date entries
    current_df.drop_duplicates(subset="date", inplace=True, keep="last")

    # write the data back out
    current_df.to_parquet(DATA_PATH, index=False, engine="pyarrow")

    print("The data update script is complete")
    print(f"     - {record_updates} existing records updated")
    print(f"     - {n_new_records} new records updated")
