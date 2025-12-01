import torch.nn as nn

LOOKBACK = 30
N_PREDICT = 10
N_FEATURES = 2    # LAT, LON

class LSTMModel(nn.Module):
    def __init__(self,hidden_size,num_layers):
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=N_FEATURES, # number of features per timestep
            hidden_size=hidden_size, # how many hidden units the LSTM should use
            num_layers=num_layers, # number of stacked LSTM layers
            batch_first=True     # input shape: (B, 30, 5) - batch is the first dimension
        )

        self.l_out = nn.Linear(
            in_features=hidden_size,
            out_features=N_PREDICT * N_FEATURES,
            bias=False
        )
    
    def forward(self, x):
        out, _ = self.lstm(x)                    # out shape: (batch, LOOKBACK, hidden)
        last_output = out[:, -1, :]              # takes the output of the last time step. shape: (batch, hidden)
        y = self.l_out(last_output)              # shape: (batch, N_PREDICT * N_FEATURES)
        return y.view(-1, N_PREDICT, N_FEATURES) # reshape to (batch, N_PREDICT, N_FEATURES)
