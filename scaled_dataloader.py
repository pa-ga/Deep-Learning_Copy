from collections.abc import Generator
import itertools
from typing import TypeVar
import pandas as pd
import numpy as np
import torch
import tqdm
from torch.utils.data import Dataset
import pyarrow.parquet as pq
from paths import SPLITS_DIR
from sklearn.preprocessing import StandardScaler

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

def to_tensors(df: pd.DataFrame) -> Pair[torch.Tensor]:
    """Extracts features from a DataFrame
    
    The DataFrame should have a row for each Segment (as produced by preprocessing.py).
    There must be no missing values.
    Each Segment is converted into a variable number of observations using sliding_window.
    Then the timestamps are replaced by a positional encoding: the first observation for each segment gets index 0 and so forth.
    
    The resulting train tensor has dimensions n_windows x 30 x 4.
    The first dimension indexes over the sliding windows, the second over the sequence of 30 data points and the third over the 4 features of each data point (LAT, LONG, SOG, COG).
    """
    # sort by ts so we can easily compute the positional encoding
    df.sort_values(by='Timestamp')
    df.drop(columns=['Timestamp', 'MMSI'], inplace=True)
    segments = df.groupby('segment_id').apply(pd.DataFrame.to_numpy, include_groups=False).to_numpy()
    # Now we have a Numpy array of length (n_segments) in the first layer, (segment length) in the second layer
    print("Initial data manipulation done, computing sliding windows and building x and y tensors . . .")
    windows = itertools.chain.from_iterable(sliding_windows(segment) for segment in tqdm.tqdm(segments))
    xs = []
    ys = []
    for x, y in windows:
        xs.append(torch.Tensor(x))
        ys.append(torch.Tensor(y))
    print("Built tensor lists")
    xs = torch.stack(xs)
    ys = torch.stack(ys)
    print("Built full tensors")
    print(xs.shape, ys.shape)
    return xs, ys

# dataset with scaling + COG sin/cos encoding
class AisDataset(Dataset): 
    def __init__(self, x_tensor: Tensor, y_tensor: Tensor, scaler: StandardScaler | None = None):
        """
        x_tensor: raw shape (n_windows, 30, 4)
        y_tensor: raw shape (n_windows, 10, 4)
        scaler: fitted StandardScaler (train) or same scaler (val)
        """
        self.scaler = scaler

        self.x = self._process(x_tensor)
        self.y = self._process(y_tensor)

    def _process(self, tensor: Tensor) -> Tensor:
        """
        Takes in a tensor of shape (N, seq, 4) and returns (N, seq, 5):
        - LAT, LON, SOG scaled with StandardScaler
        - COG → sin/cos encoding
        """
        arr = tensor.numpy()

    
        # keep only LAT, LON → shape (N, seq_len, 2)
        arr = np.concatenate(
            [arr[:, :, :2]],
            axis=-1
        )

        if self.scaler is not None:
            N, seq_len, F = arr.shape      # F = 2
            flat = arr.reshape(-1, F)      # (N*seq_len, 2)
            flat = self.scaler.transform(flat)
            arr = flat.reshape(N, seq_len, F)

        return torch.tensor(arr, dtype=torch.float32)


    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# raw tensor loaders (for fitting scaler)
def _load_raw(split_name: str) -> Pair[Tensor]:
    df = pq.read_table(SPLITS_DIR / f"{split_name}.parquet").to_pandas()
    return to_tensors(df)


# public API for use in other files
def load_train() -> tuple[AisDataset, StandardScaler]:
    print("Loading TRAIN...")
    x_train, y_train = _load_raw("train")

    # fit scaler on LAT/LON/SOG only
    flat_train = x_train.numpy().reshape(-1, 3)
    scaler = StandardScaler()
    scaler.fit(flat_train[:, :2])

    ds = AisDataset(x_train, y_train, scaler=scaler)
    return ds, scaler

def load_val(scaler) -> AisDataset:
    print("Loading VAL...")
    x_val, y_val = _load_raw("val")
    return AisDataset(x_val, y_val, scaler=scaler)


# just for testing
if __name__ == '__main__':
    train_ds, scaler = load_train()
    val_ds = load_val(scaler)
    print("Train windows:", len(train_ds))
    print("Val windows:", len(val_ds))
