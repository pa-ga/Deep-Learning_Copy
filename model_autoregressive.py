import torch
import torch.nn as nn

N_FEATURES = 2  # LAT, LON
LOOKBACK = 30
N_PREDICT = 10

class Seq2SeqLSTM(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(
            input_size=N_FEATURES,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(
            input_size=N_FEATURES,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Linear output
        self.l_out = nn.Linear(hidden_size, N_FEATURES)
    
    def forward(self, x):        
        # -------- Encoder --------
        _, (h, c) = self.encoder_lstm(x)  # h,c shape: (num_layers, batch, hidden)
        
        # -------- Decoder --------
        # Start with last input timestep as first decoder input
        decoder_input = x[:, -1, :].unsqueeze(1)  # shape: (batch, 1, N_FEATURES)
        outputs = []
        
        for _ in range(N_PREDICT):
            out, (h, c) = self.decoder_lstm(decoder_input, (h, c))
            y = self.l_out(out)  # shape: (batch, 1, N_FEATURES)
            outputs.append(y)
            
            # Feed the prediction back as next input (autoregressive)
            decoder_input = y
        
        # Concatenate all predictions along time dimension
        outputs = torch.cat(outputs, dim=1)  # shape: (batch, N_PREDICT, N_FEATURES)
        return outputs
