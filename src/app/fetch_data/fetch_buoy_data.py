import pandas as pd
from modeling.utilities.utils_buoy_data import (
    collapse_to_daily,
    create_buoy_cols,
    read_buoy_texts,
)

buoy_data_path = 'https://www.ndbc.noaa.gov/data/realtime2/LJAC1.txt'

def fetch_daily_buoy_data()->pd.DataFrame:
    """
    fetches the latest buoy data for LJAC1.txt
    """

    # read the buoy data path into a df
    df = read_buoy_texts(buoy_data_path)

    # create buoy cols
    buoy_df = create_buoy_cols(df)

    # collapse to daily
    daily_df = collapse_to_daily(buoy_df)

    return daily_df