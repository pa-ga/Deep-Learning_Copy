import matplotlib.pyplot as plt
import numpy as np
import torch

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x  # already numpy

def plot_paths(x_sample, y_true, y_pred, idx, scaler):
    # Convert to numpy safely
    x_np = to_numpy(x_sample).reshape(30, 2)
    y_true_np = to_numpy(y_true).reshape(10, 2)
    y_pred_np = to_numpy(y_pred).reshape(10, 2)

    # Inverse-transform LAT/LON
    x_np = scaler.inverse_transform(x_np)
    y_true_np = scaler.inverse_transform(y_true_np)
    y_pred_np = scaler.inverse_transform(y_pred_np)

    # Plot
    plt.figure(figsize=(6,6))
    plt.plot(x_np[:,0], x_np[:,1], 'bo-', label='Past')
    plt.plot(y_true_np[:,0], y_true_np[:,1], 'go-', label='True')
    plt.plot(y_pred_np[:,0], y_pred_np[:,1], 'ro--', label='Predicted')
    plt.xlabel('Latitude')
    plt.ylabel('Longitude')
    plt.title(f'Trajectory Sample {idx}')
    plt.legend()
    plt.show()

