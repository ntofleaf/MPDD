# Copyright (c) OpenMMLab. All rights reserved.
import torch
import numpy as np


def weighted_asymmtric_ce_loss(y_pred, y_true, epsilon=1e-8):
    """
    Compute binary cross-entropy loss manually.

    Args:
    y_pred (torch.Tensor): Tensor containing predicted probabilities.
    y_true (torch.Tensor): Tensor containing actual labels (0 or 1).
    epsilon (float): Small value to ensure numerical stability.

    Returns:
    torch.Tensor: The computed binary cross-entropy loss.
    """
    rates = [77, 22, 26, 25]
    weight_deno = np.array([ 1 / rate for rate in rates]).sum()
    weights = [(1 / rate) / weight_deno for rate in rates]
    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights.unsqueeze(0).to(y_pred.device)
    # Ensure the predicted probabilities are clipped to avoid log(0)
    y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)
    # slopes = y_true[:, 4].unsqueeze(1)
    y_true = y_true[:, :4]
    # Calculate the cross entropy
    loss = - y_true * torch.log(y_pred) 
    # loss = weights * loss * slopes
    loss = weights * loss
    # Return the mean of the loss across all observations
    return torch.mean(loss)


def weighted_asymmtric_bce_loss(y_pred, y_true, epsilon=1e-8):
    """
    Compute binary cross-entropy loss manually.

    Args:
    y_pred (torch.Tensor): Tensor containing predicted probabilities.
    y_true (torch.Tensor): Tensor containing actual labels (0 or 1).
    epsilon (float): Small value to ensure numerical stability.

    Returns:
    torch.Tensor: The computed binary cross-entropy loss.
    """
    rates = [77, 22, 26, 25]
    weight_deno = np.array([ 1 / rate for rate in rates]).sum()
    weights = [(1 / rate) / weight_deno for rate in rates]
    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights.unsqueeze(0).to(y_pred.device)
    # Ensure the predicted probabilities are clipped to avoid log(0)
    y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)
    # slopes = y_true[:, 4].unsqueeze(1)
    y_true = y_true[:, :4]
    # Calculate the binary cross entropy
    loss = - y_true * torch.log(y_pred) - (1 - y_true) * torch.log(1 - y_pred)
    # loss = weights * loss * slopes
    loss = weights * loss 
    # Return the mean of the loss across all observations
    return torch.mean(loss)


def weighted_asymmtric_bce_loss_(y_pred, y_true, epsilon=1e-8):
    """
    Compute binary cross-entropy loss manually.

    Args:
    y_pred (torch.Tensor): Tensor containing predicted probabilities.
    y_true (torch.Tensor): Tensor containing actual labels (0 or 1).
    epsilon (float): Small value to ensure numerical stability.

    Returns:
    torch.Tensor: The computed binary cross-entropy loss.
    """
    rates = [77, 22, 26, 25]
    weight_deno = np.array([ 1 / rate for rate in rates]).sum()
    weights = [(1 / rate) / weight_deno for rate in rates]
    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights.unsqueeze(0).to(y_pred.device)
    # Ensure the predicted probabilities are clipped to avoid log(0)
    y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)
    
    # Calculate the binary cross entropy
    loss = - y_true * torch.log(y_pred) - (1 - y_true) * torch.log(1 - y_pred)
    loss = weights * loss
    # Return the mean of the loss across all observations
    return torch.mean(loss)

# Example usage:
# Assume we have some predictions and actual values
# y_pred = torch.tensor([[0.7, 0.2, 0.6, 0.5], [0.4, 0.9, 0.2, 0.4]], dtype=torch.float32)  # shape [2, 3]
# y_true = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=torch.float32) 

# # Calculate the loss
# loss = weighted_asymmtric_bce_loss(y_pred, y_true)
# print(f"Calculated Loss: {loss.item()}")
