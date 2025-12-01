#Transformer Notebook

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from plot_trajectory import plot_paths
#from metrics import rmse, mse, mae

class PositionalEncoding(nn.Module):
    """
    Implements the positional encoding as described in the "Attention is All You Need" paper.
    
    Largely borrowed from the lecture material.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        pe = pe.unsqueeze(0)     # → [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: [batch, seq_len, d_model]
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class TrajectoryTransformer30to10(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        output_dim: int = 2,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.output_dim = output_dim
        self.future_steps = 10

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,              
        )

        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.out = nn.Linear(d_model, output_dim * self.future_steps)

    def forward(self, src):
        """
        src: [batch, 30, input_dim]
        return: [batch, 10, output_dim]
        """
        x = self.input_proj(src)       # [B, 30, d_model]
        x = self.pos_enc(x)            # [B, 30, d_model]
        x = self.encoder(x)            # [B, 30, d_model]

        last_state = x[:, -1, :]       # [B, d_model]

        out = self.out(last_state)     # [B, 10 * output_dim]

        return out.reshape(-1, 10, self.output_dim)

    