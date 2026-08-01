import pandas as pd
import numpy as np
import urllib.request


def fix_year(year):
    if len(str(year)) == 2:
        if year < 30:
            new_year = 2000 + year
        else:
            new_year = 1900 + year
        return new_year
    return year


# ---------------------------------------------------------------------------- #


def _get_first_line(path):
    if path.startswith(("http://", "https://")):
        with urllib.request.urlopen(path) as t:
            return t.readline().decode("utf-8")
    else:
        with open(path) as t:
            return t.readline()


# ---------------------------------------------------------------------------- #


def read_buoy_texts(buoy_txt_path):
    """
    purpose: read in the buoy text files into dataframes.
    """

    # with open(buoy_txt_path) as t:
    #     first_line = t.readline()
    first_line = _get_first_line(buoy_txt_path)

    if first_line.startswith("#"):
        df = pd.read_csv(buoy_txt_path, sep=r"\s+", skiprows=[1], header=0)
        df.columns = [c.lstrip("#") for c in df.columns]

        return df
    df = pd.read_csv(buoy_txt_path, sep=r"\s+", header=0)
    df.rename(columns={"WD": "WDIR", "YYYY": "YY"}, inplace=True)
    if df.YY.max() < 100:
        df["YY"] = df.YY.apply(lambda x: fix_year(x))
    return df


# ---------------------------------------------------------------------------- #


def create_buoy_cols(buoy_df: pd.DataFrame):

    datetime_parts = {"year": buoy_df.YY, "month": buoy_df.MM, "day": buoy_df.DD}

    buoy_df["date"] = pd.to_datetime(datetime_parts)

    cols_to_keep = ["date", "hh", "WDIR", "WSPD"]

    final_buoy_df = buoy_df[cols_to_keep].copy()

    # replace the 99 readings with null
    final_buoy_df["WDIR"] = final_buoy_df.WDIR.replace({999: np.nan})
    final_buoy_df["WSPD"] = final_buoy_df.WSPD.replace({99: np.nan})

    # set types

    final_buoy_df['WDIR'] = pd.to_numeric(final_buoy_df.WDIR, errors='coerce')
    final_buoy_df['WSPD'] = pd.to_numeric(final_buoy_df.WSPD, errors='coerce')

    final_buoy_df.dropna(inplace=True)

    return final_buoy_df


# ---------------------------------------------------------------------------- #


def collapse_to_daily(buoy_df):
    daily_mean_buoy_data = (
        buoy_df.groupby(["date"])[["WSPD", "WDIR"]].mean().reset_index()
    )

    daily_mean_buoy_data['date'] = daily_mean_buoy_data.date.dt.normalize()
    daily_mean_buoy_data['date'] = daily_mean_buoy_data['date'].dt.tz_localize(None)
    return daily_mean_buoy_data
