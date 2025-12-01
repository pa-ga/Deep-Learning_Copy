#!/usr/bin/env python3

from collections.abc import Generator
import sys
from typing import Iterable
import requests
import os
import multiprocessing
from io import BytesIO
import calendar
import functools
import itertools

import pyarrow
import pyarrow.parquet
import pandas as pd

from zip_reader import read_zip
from preprocessing import preprocess_full, preprocess_partial
from paths import PROCESSED_DATA_DIR

MONTH_URL_TEMPLATE='http://aisdata.ais.dk/{year}/aisdk-{year}-{month:02d}.zip'
DAY_URL_TEMPLATE='http://aisdata.ais.dk/{year}/aisdk-{year}-{month:02d}-{day:02d}.zip'
NEWEST_DAY_URL_TEMPLATE='http://aisdata.ais.dk/aisdk-{year}-{month:02d}-{day:02d}.zip'

verbose = True
if verbose:
    debug = functools.partial(print, file=sys.stderr)
else:
    def debug(*args, **kwargs):
        pass

def fmt_year_month(year: int, month: int):
    assert year >= 1000 and 1 <= month <= 12
    return f"{year}-{month:02d}"

def get_only(it: Iterable):
    ret_val = next(it)
    assert_stopiter(it)
    return ret_val

def assert_stopiter(it):
    try:
        extra = next(it)
        raise AssertionError(f"No StopIteration encountered, got {extra}")
    except StopIteration:
        pass

def get_day(year: int, month: int, day: int) -> pd.DataFrame:
    debug(f"Getting {year}-{month}-{day}")
    resp = requests.get(DAY_URL_TEMPLATE.format(year=year, month=month, day=day))
    if resp.status_code < 400:
        df = get_only(read_zip(BytesIO(resp.content)))
        debug(f"Got {year}-{month}-{day}")
        return df
    debug(f"Got {resp.status_code} for {year}-{month}-{day}, trying the root...")
    resp = requests.get(NEWEST_DAY_URL_TEMPLATE.format(year=year, month=month, day=day))
    resp.raise_for_status()
    df = get_only(read_zip(BytesIO(resp.content)))
    debug(f"Done with {year}-{month}-{day} (from root)")
    return df

def fetch_month(year: int, month: int) -> Generator[pd.DataFrame]:
    assert year >= 2006 and 1 <= month <= 12
    month_length = calendar.monthrange(year, month)[1]
    # first try the generic month-based URL
    debug(f"Fetching {year}-{month}")
    resp = requests.get(MONTH_URL_TEMPLATE.format(year=year, month=month))
    if resp.status_code < 300:
        n = 0
        for df in read_zip(BytesIO(resp.content)):
            yield preprocess_partial(df)
            n += 1

        # assert there is one df per day
        assert n == month_length
        debug(f"Done with {year}-{month}")
        return

    debug(f"Got {resp.status_code} for {year}-{month}, fetching days...")

    # otherwise, assume we access the month on a day basis.
    yield from (preprocess_partial(get_day(year, month, day)) for day in range(1, month_length + 1))
    debug(f"Fetched all {month_length} days for month {year}-{month}")
    return

def main():
    year = 2024
    parallel = False
    if parallel:
        with multiprocessing.Pool(os.cpu_count() - 2) as pool:
            month_dfs = pool.map(functools.partial(fetch_month, year), range(1, 13))
    else:
        month_dfs = (fetch_month(year, month) for month in range(1, 13))
    debug("Starting full concat")
    df = pd.concat(itertools.chain.from_iterable(month_dfs))
    del month_dfs
    debug("End of concat, running preprocess_full()")
    df = preprocess_full(df)
    table = pyarrow.Table.from_pandas(df, preserve_index=False)
    pyarrow.parquet.write_to_dataset(table, root_path=PROCESSED_DATA_DIR, partition_cols=['MMSI', 'Segment'])

if __name__ == '__main__':
    main()
