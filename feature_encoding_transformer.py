import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

class LatLonPreprocessor:
    def __init__(self, train_ds, scaler, embed_dim=128):
        """
        Preprocess lat and lon into embeddings and bin indices.
        Args:
            train_ds: original dataset (scaled tensors)
            scaler: fitted scaler used for inverse transform
        """
        self.scaler = scaler
        self.bin_size_lat = 0.01
        self.bin_size_lon = 0.01

        # Compute bounds
        lat_min, lat_max = float("inf"), float("-inf")
        lon_min, lon_max = float("inf"), float("-inf")

        for x, _ in train_ds:
            arr = scaler.inverse_transform(x[:, :2].cpu().numpy())
            lat_min = min(lat_min, arr[:,0].min())
            lat_max = max(lat_max, arr[:,0].max())
            lon_min = min(lon_min, arr[:,1].min())
            lon_max = max(lon_max, arr[:,1].max())

        self.LAT_MIN = lat_min - 1
        self.LAT_MAX = lat_max + 1
        self.LON_MIN = lon_min - 1
        self.LON_MAX = lon_max + 1

        self.num_lat_bins = int((self.LAT_MAX - self.LAT_MIN) / self.bin_size_lat)
        self.num_lon_bins = int((self.LON_MAX - self.LON_MIN) / self.bin_size_lon)

        # embeddings sized by training bins
        self.lat_embedding = nn.Embedding(self.num_lat_bins, embed_dim)
        self.lon_embedding = nn.Embedding(self.num_lon_bins, embed_dim)
        self.embed_dim = embed_dim

    def _encode_dataset(self, dataset):
        """Encode any dataset (train/val/test) using training bins/embeddings."""
        input_emb_list, lat_targets_list, lon_targets_list = [], [], []

        for x, y in dataset:
            # inverse transform lat, lon, sog
            x_np = self.scaler.inverse_transform(x[:, :3].cpu().numpy())
            y_np = self.scaler.inverse_transform(y[:, :3].cpu().numpy())

            x_unscaled = torch.tensor(x_np)
            y_unscaled = torch.tensor(y_np)

            lat_inputs, lon_inputs = x_unscaled[:,0], x_unscaled[:,1]
            lat_targets, lon_targets = y_unscaled[:,0], y_unscaled[:,1]

            # bin indices
            lat_id_inputs = ((lat_inputs - self.LAT_MIN) / self.bin_size_lat).floor().long().clamp(0, self.num_lat_bins-1)
            lon_id_inputs = ((lon_inputs - self.LON_MIN) / self.bin_size_lon).floor().long().clamp(0, self.num_lon_bins-1)

            lat_id_targets = ((lat_targets - self.LAT_MIN) / self.bin_size_lat).floor().long().clamp(0, self.num_lat_bins-1)
            lon_id_targets = ((lon_targets - self.LON_MIN) / self.bin_size_lon).floor().long().clamp(0, self.num_lon_bins-1)

            # embeddings
            lat_emb = self.lat_embedding(lat_id_inputs)
            lon_emb = self.lon_embedding(lon_id_inputs)
            input_emb = torch.cat([lat_emb, lon_emb], dim=-1)

            input_emb_list.append(input_emb)
            lat_targets_list.append(lat_id_targets)
            lon_targets_list.append(lon_id_targets)

        # stack into tensors
        input_emb = torch.stack(input_emb_list)
        lat_targets = torch.stack(lat_targets_list)
        lon_targets = torch.stack(lon_targets_list)

        return TensorDataset(input_emb, lat_targets, lon_targets)

    def preprocess(self, train_ds):
        return self._encode_dataset(train_ds)

    def preprocess_val(self, val_ds):
        return self._encode_dataset(val_ds)

    def preprocess_test(self, test_ds):
        return self._encode_dataset(test_ds)
