import math
import numpy as np
from sklearn.linear_model import LinearRegression
import torch

LinearModel = LinearRegression

class LinearModel(LinearRegression):
    def __init__(self):
        super().__init__()

    # custom training and evaluation for linear model since it does not
    # use gradients, optimizers or batches
    @classmethod
    def train(cls, train: torch.Tensor):
        model = cls()
        train_x = train.x.view(train.x.size(0), -1)
        train_y = train.y.view(train.y.size(0), -1)

        model.fit(train_x, train_y)
        return model

    def evaluate(self, train: torch.Tensor, val: torch.Tensor):
        train_x = train.x.view(train.x.size(0), -1)
        train_y = train.y.view(train.y.size(0), -1)
        val_x = val.x.view(val.x.size(0), -1)
        val_y = val.y.view(val.y.size(0), -1)

        y_train_pred = self.predict(train_x)
        y_val_pred = self.predict(val_x)
        
        train_mse = ((y_train_pred - train_y.numpy()) ** 2).mean().item()
        train_rmse = math.sqrt(train_mse)
        train_mae = (np.abs(y_train_pred - train_y.numpy())).mean().item()
        train_loss = train_mse
        val_mse = ((y_val_pred - val_y.numpy()) ** 2).mean().item()
        val_rmse = math.sqrt(val_mse)
        val_mae = (np.abs(y_val_pred - val_y.numpy())).mean().item()
        val_loss = val_mse

        return {
            'train_loss': train_loss,
            'train_mse': train_mse,
            'train_rmse': train_rmse,
            'train_mae': train_mae,
            'val_loss': val_loss,
            'val_mse': val_mse,
            'val_rmse': val_rmse,
            'val_mae': val_mae,
        }
