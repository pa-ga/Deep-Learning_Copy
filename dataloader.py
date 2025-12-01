from collections.abc import Generator
import itertools
from typing import Optional, TypeVar
import pandas as pd
import numpy as np
import torch
import tqdm
from torch.utils.data import Dataset
import pyarrow.parquet as pq
from paths import SPLITS_DIR
from sklearn.preprocessing import StandardScaler
import joblib

LOOKBACK = 30
N_PREDICT = 10

A = TypeVar('A')
Pair = tuple[A, A]
Tensor = torch.Tensor

def sliding_windows(segment: np.array) -> Generator[Pair[np.array]]:
    """Split a segment into sliding window rows.
    
    Uses the following rule:
    Output observations 0 through 29 as x, then 30 through 39 as y.
    Output obs. 10 through 39 as x, then 40 through 49 as y.
    ...
    As soon as there is not enough values to output an observation, stop.
    This means that some observations may not be used.
    """

    full_batches = (len(segment) - LOOKBACK) // N_PREDICT
    for i in range(full_batches):
        x_start = N_PREDICT*i
        y_start = x_start + LOOKBACK
        y_end = y_start + N_PREDICT
        yield segment[x_start:y_start], segment[y_start:y_end]

def is_stationary_segment(segment, threshold = 2e-4):
    """Detects whether a vessel is moving or not.
    threshold: lat and lon movement below threshold is considered stationary
    """
    lat = segment[:, 0]
    lon = segment[:, 1]

    lat_diff = lat.max() - lat.min()
    lon_diff = lon.max() - lon.min()

    return lat_diff < threshold and lon_diff < threshold


def to_tensors(df: pd.DataFrame, filter_stationary: bool) -> Pair[torch.Tensor]:
    """Extracts features from a DataFrame
    
    The DataFrame should have a row for each Segment (as produced by preprocessing.py).
    There must be no missing values.
    Each Segment is converted into a variable number of observations using sliding_window.
    Then the timestamps are replaced by a positional encoding: the first observation for each segment gets index 0 and so forth.
    
    The resulting train tensor has dimensions n_windows x 30 x 4.
    The first dimension indexes over the sliding windows, the second over the sequence of 30 data points and the third over the 4 features of each data point (LAT, LONG, SOG, COG).
    """
    # sort by ts so we can easily compute the positional encoding
    df.sort_values(by='Timestamp', inplace=True)
    df.drop(columns=['Timestamp', 'MMSI', 'SOG', 'COG'], inplace=True)
    segments = df.groupby('segment_id').apply(pd.DataFrame.to_numpy, include_groups=False).to_numpy()
    # Now we have a Numpy array of length (n_segments) in the first layer, (segment length) in the second layer
    windows = itertools.chain.from_iterable(sliding_windows(segment) for segment in tqdm.tqdm(segments))
    xs = []
    ys = []
    for x, y in windows:
        if not filter_stationary or (not is_stationary_segment(x) and not is_stationary_segment(y)):
            xs.append(torch.Tensor(x))
            ys.append(torch.Tensor(y))
    xs = torch.stack(xs)
    ys = torch.stack(ys)
    return xs, ys

# dataset with scaling
class AisDataset(Dataset): 
    scaler: StandardScaler
    
    def __init__(self, x_tensor: Tensor, y_tensor: Tensor, scaler: Optional[StandardScaler] = None):
        """
        x_tensor: raw shape (n_windows, 30, 2)
        y_tensor: raw shape (n_windows, 10, 2)
        scaler: fitted StandardScaler (train) or same scaler (val)
        """
        self.scaler = scaler

        self.x = self._process(x_tensor)
        self.y = self._process(y_tensor)

    def _process(self, tensor: Tensor) -> Tensor:
        """
        Takes in a tensor of shape (N, seq, 2) and returns (N, seq, 2):
        - LAT, LON scaled with StandardScaler
        """
        arr = tensor.numpy()

        if self.scaler is not None:
            N, seq_len, n_features = arr.shape      # n_features = 2
            flat = arr.reshape(N * seq_len, n_features)
            flat = self.scaler.transform(flat)
            arr = flat.reshape(N, seq_len, n_features)

        return torch.tensor(arr, dtype=torch.float32)


    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# raw tensor loaders (for fitting scaler)
def _load_raw(split_name: str, filter_stationary: bool) -> Pair[Tensor]:
    df = pq.read_table(SPLITS_DIR / f"{split_name}.parquet").to_pandas()
    return to_tensors(df, filter_stationary)


# public API for use in other files
def load_train(filter_stationary: bool = True) -> tuple[AisDataset, StandardScaler]:
    print("Loading TRAIN...")
    x_train, y_train = _load_raw("train", filter_stationary)

    # fit scaler on LAT/LON only
    scaler = StandardScaler()
    N, seq_len, n_features = x_train.shape
    flat_train = x_train.reshape(N * seq_len, n_features)
    scaler = StandardScaler()
    scaler.fit(flat_train[:, :2])

    ds = AisDataset(x_train, y_train, scaler=scaler)
    return ds, scaler

def load_val(scaler: StandardScaler, filter_stationary: bool = True) -> AisDataset:
    print("Loading VAL...")
    x_val, y_val = _load_raw("val", filter_stationary)
    return AisDataset(x_val, y_val, scaler=scaler)


if __name__ == '__main__':
    # save scaler for later use
    train_ds, scaler = load_train(filter_stationary=True)
    joblib.dump(scaler, "scaler_filtered.save")
    train_ds, scaler = load_train(filter_stationary=False)
    joblib.dump(scaler, "scaler_unfiltered.save")

