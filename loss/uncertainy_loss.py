import torch
import torch.nn as nn

class uc_loss(nn.Module):
    def __init__(self):
        super(uc_loss, self).__init__()

    def forward(self, gt, mu, theta):
        return torch.mean((gt - mu) ** 2 / (1e-6 + 2 * torch.exp(theta)) + theta)