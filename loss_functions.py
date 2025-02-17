# loss_functions.py

import torch
import torch.nn as nn
from torch.nn import functional as F

def asymmetric_mse_loss(output, target, smaller_weight=1.5, larger_weight=0.5):
    """
    Asymmetric MSE loss which penalizes more heavily when the output pixel value
    is smaller than the target pixel value.
    """
    smaller = (output < target).float()
    larger_or_equal = (output >= target).float()
    loss = (smaller_weight * smaller * (target - output) ** 2 +
            larger_weight * larger_or_equal * (output - target) ** 2)
    return loss.mean()

mse_loss = nn.MSELoss()

def compute_loss_regular(x, x_hat1, sentence_length):
    """
    Mode 1: Regular loss.
    Always select the final (max-length) output from x_hat1.
    """
    # Create a tensor specifying the max index
    max_idx = torch.full_like(x[:, 0, 0, 0], sentence_length - 1, dtype=torch.long)
    max_idx_expanded = max_idx.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    # Gather the appropriate slice
    x_hat = torch.gather(
        x_hat1, 
        1, 
        max_idx_expanded.expand(-1, -1, x.size(1), x.size(2), x.size(3))
    ).squeeze(1)
    
    recon_loss = mse_loss(x_hat, x)
    return recon_loss


def compute_loss_progressive(x, x_hat1, ica_orders):
    """
    Mode 2: Progressive loss.
    We choose from x_hat1 according to the prefix length = ica_orders for each sample.
    """
    ica_orders_expanded = ica_orders.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    x_hat = torch.gather(
        x_hat1, 
        1, 
        ica_orders_expanded.expand(-1, -1, x.size(1), x.size(2), x.size(3))
    ).squeeze(1)

    recon_loss = mse_loss(x_hat, x)
    return recon_loss


def compute_loss_progressive_strict(x, x_hat1, ica_orders):
    """
    Mode 3: Progressive Strict.
    For each sample, compute MSE for *all* images up to that prefix index.
    """
    losses_batch = []
    for i in range(x.size(0)):
        prefix_len = ica_orders[i].item()  # e.g., number of symbols used
        if prefix_len < 1:
            prefix_len = 1  # Safety in case it's 0 or negative
        
        # Expand x[i] -> replicate it prefix_len times
        expanded_x = x[i].unsqueeze(0).expand(prefix_len, -1, -1, -1)
        truncated_x_hat = x_hat1[i, :prefix_len]  # shape: [prefix_len, C, H, W]
        
        individual_loss = mse_loss(truncated_x_hat, expanded_x)
        losses_batch.append(individual_loss)
    
    recon_loss = torch.stack(losses_batch).mean()
    return recon_loss


def compute_loss_progressive_strict_containing_bias(x, x_hat1, ica_orders):
    """
    Mode 4: Progressive Strict + Containing Bias.
    Same as Progressive Strict, but uses asymmetric_mse_loss.
    """
    losses_batch = []
    for i in range(x.size(0)):
        prefix_len = ica_orders[i].item()
        if prefix_len < 1:
            prefix_len = 1
        
        expanded_x = x[i].unsqueeze(0).expand(prefix_len, -1, -1, -1)
        truncated_x_hat = x_hat1[i, :prefix_len]
        
        individual_loss = asymmetric_mse_loss(truncated_x_hat, expanded_x)
        losses_batch.append(individual_loss)

    recon_loss = torch.stack(losses_batch).mean()
    return recon_loss


def compute_loss_dispatcher(x, x_hat1, ica_orders, loss_mode, sentence_length):
    """
    Master function that dispatches to the appropriate loss mode.
    """
    if loss_mode == 1:
        return compute_loss_regular(x, x_hat1, sentence_length)
    elif loss_mode == 2:
        return compute_loss_progressive(x, x_hat1, ica_orders)
    elif loss_mode == 3:
        return compute_loss_progressive_strict(x, x_hat1, ica_orders)
    elif loss_mode == 4:
        return compute_loss_progressive_strict_containing_bias(x, x_hat1, ica_orders)
    else:
        raise ValueError(f"Invalid loss_mode={loss_mode}. Must be 1,2,3,4.")
