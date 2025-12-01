from pathlib import Path
from collections.abc import Generator
from typing import IO, Union
from zipfile import ZipFile
import pandas as pd
import os

DTYPES = {
        "MMSI": "object",
        "SOG": float,
        "COG": float,
        "Longitude": float,
        "Latitude": float,
        "# Timestamp": "object",
        "Type of mobile": "object",
}

def read_zip(p: Union[os.PathLike, IO[bytes]]) -> Generator[pd.DataFrame]:
    with ZipFile(p) as zip:
        files = sorted((f for f in zip.infolist() if f.filename.endswith('.csv')), key=lambda f: f.filename)
        for file in files:
            with zip.open(file) as fp:
                usecols = list(DTYPES.keys())
                yield pd.read_csv(fp, usecols=usecols, dtype=DTYPES)

def read_all(dir: Path) -> Generator[pd.DataFrame]:
    zips = sorted(filter(lambda p: p.suffix == '.zip', dir.iterdir()), key=lambda p: p.name)
    for day_f in zips:
        yield from read_zip(day_f)
