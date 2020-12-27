import numpy as np
import torch
import torch.nn as nn


class NoteDropout(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        ones = np.ones(input.shape)
        zeros = np.zeros(input.shape)
        mask = np.random.random_sample(input.shape)
        mask = np.where(mask <= p.item(), ones, zeros)
        mask = torch.tensor(mask, dtype=torch.float, device=input.device)
        return torch.mul(input, mask)
