"""Preprocessing functionality.

The data set is read one day at a time. Some preprocessing is
done one the DataFrame for the single day. When each day has been
preprocessed, it is added to an aggregate DataFrame, which is then
preprocessed again in the end.
"""

import pandas as pd

BBOX = [60, 0, 50, 20]
KNOTS_TO_MS = 0.514444

def track_filter(g):
        len_filt = len(g) > 256  # Min required length of track/segment
        sog_filt = 1 <= g["SOG"].max() <= 50  # Remove stationary tracks/segments
        time_filt = (g["Timestamp"].max() - g["Timestamp"].min()).total_seconds() >= 60 * 60  # Min required timespan
        return len_filt and sog_filt and time_filt

def downsample(df: pd.DataFrame, resolution):
     df['truncated'] = df['Timestamp'].dt.floor(resolution)
     df.drop_duplicates(['truncated', 'MMSI'], keep='first', inplace=True)
     df.drop(columns= ['truncated'], inplace=True) 

def preprocess_partial(df: pd.DataFrame) -> pd.DataFrame:
    """This preprocessing is applied to each day-df individually"""
    # Remove location errors by filtering out points outside bounding box
    north, west, south, east = BBOX
    df = df[(df["Latitude"] <= north) & (df["Latitude"] >= south) & (df["Longitude"] >= west) & (
            df["Longitude"] <= east)]
    
    # Keeps only Class A and Class B ships. Drops “Type of mobile” column afterward.
    df = df[df["Type of mobile"].isin(["Class A", "Class B"])].drop(columns=["Type of mobile"])
    df = df[df["MMSI"].str.len() == 9]  # Adhere to MMSI format
    df = df[df["MMSI"].str[:3].astype(int).between(200, 775)]  # Must start with a valid Maritime Identification Digit (MID) between 200–775.

    df = df.rename(columns={"# Timestamp": "Timestamp"})
    # raises errors as opposed to Peder's code so we see any data quality issues
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors="raise")


    #df.drop_duplicates(["Timestamp", "MMSI", ], keep="first", inplace=True)
    downsample(df, '5min') #downsample to 5 minute resolution

    # Track filtering
    df = df.groupby("MMSI").filter(track_filter)
    df = df.sort_values(['MMSI', 'Timestamp']) # Sort after filtering to avoid unnecessary sorting of dropped data
    df["SOG"] = KNOTS_TO_MS * df["SOG"] # Convert SOG to m/s
    return df

def preprocess_full(df: pd.DataFrame) -> pd.DataFrame:
      # Divide track into segments based on timegap
    df['Segment'] = df.groupby('MMSI')['Timestamp'].transform(
        lambda x: (x.diff().dt.total_seconds().fillna(0) >= 15 * 60).cumsum())  # Max allowed timegap

    # Segment filtering
    df = df.groupby(["MMSI", "Segment"]).filter(track_filter)
    df.reset_index(drop=True, inplace=True)
    return df
