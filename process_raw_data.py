#!/usr/bin/env python3

"""This script loads all zip files from RAW_DATA_DIR, does preprocessing and writes the result as parquet files into PROCESSED_DATA_DIR"""

import pandas as pd
import pyarrow
import pyarrow.parquet

from zip_reader import read_all
from paths import RAW_DATA_DIR, PROCESSED_DATA_DIR
from preprocessing import preprocess_full, preprocess_partial

# In-memory size: 10.5 Gb
# single-threaded, all synchronous: 29 minutes
df: pd.DataFrame = None
for partial_df in read_all(RAW_DATA_DIR):
    partial_df = preprocess_partial(partial_df)
    if df is None:
        df = partial_df
    else:
        df = pd.concat((df, partial_df))
df = preprocess_full(df)

table = pyarrow.Table.from_pandas(df, preserve_index=False)
pyarrow.parquet.write_to_dataset(table, root_path=PROCESSED_DATA_DIR, partition_cols=['MMSI', 'Segment'])

