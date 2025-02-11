import torch
import torch.nn as nn

class Quantization_Loss(nn.Module):

    def __init__(self):
        super(Quantization_Loss, self).__init__()

    def forward(self, inputs):
        loss = -(inputs * torch.log(inputs + 1e-20) +
                 (1.0 - inputs) * torch.log(1.0 - inputs + 1e-20))
        return loss.mean()