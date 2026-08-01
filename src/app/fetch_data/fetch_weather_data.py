import io
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests

KSAN_URL = "https://forecast.weather.gov/data/obhistory/KSAN.html"

def read_ksan_obhistory(url=KSAN_URL):
    headers = {"User-Agent": "Mozilla/5.0 (compatible; research script)"}
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()

    tables = pd.read_html(io.StringIO(resp.text))
    # This page has one main data table; find the one with the most rows
    df = max(tables, key=len)

    return df

# ---------------------------------------------------------------------------- #


def add_full_date(df, as_of=None, date_col="Date", time_col="Time (pdt)"):
    """
    Reconstructs full dates for the obhistory table, which only shows day-of-month.
    Assumes df is ordered newest-first (as scraped from the page).

    as_of: datetime representing "today" at scrape time. Defaults to now.
    """
    df = df.copy()
    if as_of is None:
        # setting this to always check grab PST bc the 
        # loc
        as_of = datetime.now(tz=ZoneInfo("America/Los_Angeles"))

    current_year = as_of.year
    current_month = as_of.month

    full_dates = []
    prev_day = None

    for day in df[date_col].astype(int):
        if prev_day is not None and day > prev_day:
            # Day number went UP while moving backward in time -> month rolled back
            current_month -= 1
            if current_month == 0:
                current_month = 12
                current_year -= 1
        full_dates.append(datetime(current_year, current_month, day))
        prev_day = day

    df["FullDate"] = full_dates

    # Optional: combine with the time column into a full timestamp
    if time_col in df.columns:
        df["Timestamp"] = pd.to_datetime(
            df["FullDate"].dt.strftime("%Y-%m-%d") + " " + df[time_col],
            errors="coerce"
        )

    return df


def fetch_daily_weather()->pd.DataFrame:
    # retrieve daily weather
    df = read_ksan_obhistory()

    # set the columns to something more readable
    df.columns = [
       "Date", "Time", "Wind", "Vis", "Weather", "Sky",
       "TempAir", "TempDwpt", "Temp6hrMax", "Temp6hrMin",
       "RH", "WindChill", "HeatIndex", "Altimeter", "SeaLevelPressure",
       "Precip1hr", "Precip3hr", "Precip6hr"
   ]
    # drop the second header on the bottom of the table
    df = df[df["Date"] != "Date"].reset_index(drop=True)

    # create the full date column
    # this step depends on your machines internal clock
    # Mine is in PST so this is not an issue. However
    df = add_full_date(df)

    df['Precip1hr'] = df.Precip1hr.fillna(0)

    daily_weather = df.groupby('FullDate').TempAir.agg(['min', 'max']).reset_index()
    daily_precip_df = df.groupby('FullDate').Precip1hr.agg(['sum']).reset_index()

    daily_precip_df.rename(columns={'sum':'PRCP'}, inplace=True)

    daily_weather = daily_weather.merge(daily_precip_df,
                    how='left',
                    on='FullDate')
    daily_weather.rename(columns={'FullDate': 'date'}, inplace=True)
    daily_weather['date'] = daily_weather.date.dt.normalize()
    daily_weather['date'] = daily_weather['date'].dt.tz_localize(None)

    return daily_weather

