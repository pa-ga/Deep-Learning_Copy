import torch


transformer_defaults = {
    "optimizer": torch.optim.AdamW,
    "optimizer_args": {
        "lr": 1e-3,
        "weight_decay": 1e-4,
    },
    "batch_size": 128,
}
transformer_configs = [
    {   
        "name": "mini_transformer",
        "epochs": 10,
        "model_kwargs": {
            "d_model": 128,
            "nhead": 4,
            "num_layers": 3,
            "dim_feedforward": 10,
            "dropout": 0.1,
        },
    },
    {
        "name": "small_transformer",
        "epochs": 40,
        "model_kwargs": {
            "d_model": 128,
            "nhead": 4,
            "num_layers": 3,
            "dim_feedforward": 512,
            "dropout": 0.1,
        },
    },
    {
        "name": "medium_transformer",
        "epochs": 40,
        "model_kwargs": {
            "d_model": 256,
            "nhead": 8,
            "num_layers": 3,
            "dim_feedforward": 1024,
            "dropout": 0.1,
        },
        "optimizer_args": {
            "weight_decay": 1e-4,
        },
    },
    {
        "name": "deeper_transformer_2",
        "epochs": 90,
        "model_kwargs": {
            "d_model": 256,
            "nhead": 8,
            "num_layers": 4,
            "dim_feedforward": 1024,
            "dropout": 0.1,
        },
        "optimizer_args": {
            "weight_decay": 1e-4,
        },
    },
    {
        "name": "deeper_transformer",
        "epochs": 50,
        "model_kwargs": {
            "d_model": 256,
            "nhead": 8,
            "num_layers": 4,
            "dim_feedforward": 1024,
            "dropout": 0.1,
        },
        "optimizer_args": {
            "lr": 5e-4,
        },
    },
]

lstm_defaults = dict(batch_size=512)
lstm_configs = [
    {   
        "name": "mini_lstm",
        "epochs": 10,
        "model_kwargs": {
            "hidden_size": 64,
            "num_layers": 2,
        },
    },
    {
        "name": "small_lstm",
        "epochs": 40,
        "model_kwargs": {
            "hidden_size": 128,
            "num_layers": 3,
        },
    },
    {
        "name": "medium_lstm",
        "epochs": 40,
        "model_kwargs": {
            "hidden_size": 512,
            "num_layers": 3,
        },
        "optimizer_args": {
            "weight_decay": 1e-4,
        },
    },
    {
        "name": "deeper_lstm_2",
        "epochs": 90,
        "model_kwargs": {
            "hidden_size": 512,
            "num_layers": 4,
        },
        "optimizer_args": {
            "weight_decay": 1e-4,
        },
    },
    {
        "name": "deeper_lstm",
        "epochs": 90,
        "model_kwargs": {
            "hidden_size": 512,
            "num_layers": 5,
        },
        "optimizer_args": {
            "lr": 5e-4,
        },
    },
]

autoreg_defaults = {
    "optimizer": torch.optim.Adam,
    "optimizer_args": {
        "lr": 1e-3,
        "weight_decay": 0,
    }
}

autoreg_configs = [
    {   
        "name": "mini_autoreg_lstm",
        "epochs": 10,
        "model_kwargs": {
            "hidden_size": 64,
            "num_layers": 2,
        },
    },
    {
        "name": "small_autoreg_lstm",
        "epochs": 40,
        "model_kwargs": {
            "hidden_size": 128,
            "num_layers": 3,
        },
    },
    {
        "name": "medium_autoreg_lstm",
        "epochs": 40,
        "model_kwargs": {
            "hidden_size": 512,
            "num_layers": 3,
        },
        "optimizer_args": {
            "weight_decay": 1e-4,
        },
    },
    {
        "name": "deeper_autoreg_lstm_2",
        "epochs": 90,
        "model_kwargs": {
            "hidden_size": 512,
            "num_layers": 4,
        },
        "optimizer_args": {
            "weight_decay": 1e-4,
        },
    },
    {
        "name": "deeper_autoreg_lstm",
        "epochs": 90,
        "model_kwargs": {
            "hidden_size": 512,
            "num_layers": 5,
        },
        "optimizer_args": {
            "lr": 5e-4,
        },
    },
]
